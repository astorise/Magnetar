# kernel-execution-plan Specification

## Purpose
TBD - created by archiving change define-kernel-execution-plan-and-prepared-graph-contract. Update Purpose after archive.
## Requirements
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

### Requirement: Plan May Precompute Stream Assignment

Prepared Execution Plan SHALL be able to bind nodes or segments to logical execution stream
classes.

#### Scenario: Attention decode

Given Plan uses compute and transfer operations

When Plan is built

Then stream assignments may be determined before token execution.

### Requirement: Plan May Precompute Dependency Edges

Static execution dependencies SHALL be materializable in Prepared Execution Plan.

#### Scenario: MatMul feeds RMSNorm

Given graph topology establishes dependency

When Plan is prepared

Then dependency does not need full rediscovery on every execution.

### Requirement: Plan Does Not Store Native Streams

Prepared Execution Plan SHALL not contain Provider-native synchronization
objects.

#### Scenario: CUDA stream exists

Given Provider created native stream

When Plan is serialized or inspected

Then only logical ExecutionStream binding is present.

### Requirement: Plan Supports Dynamic Dependency Slots

Prepared Plan SHALL be able to expose dynamic dependency slots for
Session/invocation state.

#### Scenario: Prior KV update

Given decode Plan is reused across Sessions

When Session executes next step

Then its current KV CompletionToken can be bound dynamically.

### Requirement: Prepared Segment Has Logical Completion

Provider-prepared segment SHALL expose logical CompletionToken after
submission.

#### Scenario: CUDA Graph launch

Given Provider launches captured graph

When submitted

Then Runtime receives one logical completion scope without observing internal
CUDA events.

### Requirement: Plan Generation Retirement Includes Stream Work

Prepared Plan SHALL remain alive while submissions using its stream/segment
bindings remain in-flight.

#### Scenario: Plan replacement

Given generation 10 is retiring

And one submission remains pending

When generation 11 becomes active

Then generation 10 resources are not destroyed prematurely.

### Requirement: Prepared Plan May Encode Residency Assumptions

Prepared Execution Plan SHALL be able to bind Kernel inputs/outputs to required or preferred
MemoryDomains.

#### Scenario: Decode Plan

Given all decode Kernels run on GPU0

When Plan is built

Then weights, KV, intermediates, and workspace may be planned as GPU0 resident.

### Requirement: Prepared Plan May Elide Redundant Transfers

Plan construction SHALL remove movement that is unnecessary under validated
residency.

#### Scenario: Already-resident Tensor

Given previous node output is GPU0-local

And next Kernel executes on GPU0

When Plan is prepared

Then no GPU0-to-GPU0 staging copy is emitted.

### Requirement: Residency Guard

Prepared Plan SHALL not execute against Resources violating hard residency
assumptions.

#### Scenario: KV spilled to host

Given Plan requires GPU-resident KV

When decode starts

Then Runtime rebinds/transfers/replans according to policy before Kernel
execution.

### Requirement: Prepared Plan Does Not Store Native Addresses

Plan SHALL refer to logical Resource bindings rather than Device pointers.

#### Scenario: CUDA graph-related Plan

Given Provider has native addresses internally

When Plan metadata is inspected

Then raw addresses are absent.

### Requirement: Host Mapping Is An Explicit Boundary

Prepared Plan SHALL designate host-visible output boundary when host access is
required.

#### Scenario: Final logits

Given sampler executes on host

When Plan completes Device logits

Then explicit map/transfer step makes required data host-visible.

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

### Requirement: Prepared Plan Captures Exact Device Bindings

PreparedExecutionPlan SHALL identify concrete Provider/Device for every
multi-Device segment.

#### Scenario: Pipeline Plan

Given stage 0 uses GPU0 and stage 1 uses GPU1

When Plan becomes ready

Then these bindings are explicit and generation-stable.

### Requirement: Prepared Plan Captures Movement Edges

Cross-Device Resource transitions SHALL be present in the prepared execution
strategy.

#### Scenario: Activation transfer

Given GPU0 output feeds GPU1 input

When Plan is prepared

Then transfer/peer-access edge exists between stages.

### Requirement: Prepared Plan Binds Per Device Allocation Plans

A multi-Device Prepared Plan SHALL be able to reference distinct AllocationPlans for each
participating Device.

#### Scenario: Different workspace pools

Given GPU0 and GPU1 require different workspace geometry

When Plan is built

Then memory slots bind to corresponding Device pools.

### Requirement: Placement Guards Are Checked

Prepared Plan SHALL validate hard placement assumptions before use.

#### Scenario: Required peer path disappeared

Given Plan requires direct peer transfer

When capability is no longer available

Then Plan does not execute unchanged.

### Requirement: Placement Staleness Does Not Mutate Plan

A more attractive Device placement SHALL result in new Plan generation rather
than in-place binding rewrite.

#### Scenario: GPU pressure shifts

Given current Plan is still safe

When alternative placement becomes better

Then old Plan may be marked stale and replacement built.

### Requirement: Device Loss Hard Invalidates Plan

A Plan requiring lost Device SHALL receive no new work.

#### Scenario: Stage Device unavailable

Given GPU1 is lost

When new invocation tries active Plan

Then guard fails/invalidation is enforced.

### Requirement: In Flight Placement Remains Coherent

An in-flight Plan generation SHALL retain its original Device bindings until
safe completion or failure.

#### Scenario: Replacement published mid-stage

Given invocation is executing stage 0

When new Plan generation is activated

Then the invocation does not silently jump to a mixed generation.

### Requirement: First Profile Builds Prepared Execution Plan

Qwen fixture SHALL execute through PreparedExecutionPlan.

#### Scenario: Model becomes ready

Given graph and Reference CPU Kernels are available

When preparation completes

Then a ready Plan contains Kernel/resource execution decisions.

### Requirement: Single Device Plan Is Valid

First profile Plan SHALL allow all nodes to bind to one Reference CPU Device.

#### Scenario: Graph has all mandatory Operators

Given CPU provides all Kernels

When Plan is built

Then no multi-Device machinery is required.

### Requirement: Plan Contains Kernel Bindings

Ready Plan SHALL identify selected prepared Kernels rather than rediscovering
them through direct model-specific calls.

#### Scenario: Decode Plan

Given Attention/RMSNorm/MatMul bindings are prepared

When token executes

Then Plan references those bindings.

### Requirement: Decode Reuses Compatible Plan

Repeated incremental decode SHALL reuse compatible Plan generation when graph
and shape guards remain valid.

#### Scenario: Ten generated tokens

Given graph/shape guards remain valid

When tokens execute

Then complete Plan construction is not repeated for every token.

### Requirement: Prepared Plan Drives Execution
A ready PreparedExecutionPlan SHALL drive first-native dispatch through immutable PlanNodeBinding and PreparedKernelId entries without normal hot-path kernel rediscovery.

#### Scenario: Registry preference changes after publication
- **WHEN** a ready plan is executed after registry preferences change
- **THEN** execution uses the published binding from that plan generation.

#### Scenario: Bound kernel is unavailable
- **WHEN** a plan binding references a missing or revoked PreparedKernelId
- **THEN** Runtime fails with a structured plan execution error.

### Requirement: First-Native Generation Requires Prepared Plans
First-native model execution SHALL use a PreparedExecutionPlan compatible with the ModelInstance, phase, Provider, Device, dtype, layout, KV mode, and workload bucket.

#### Scenario: Compatible plan selected
- **WHEN** Runtime starts first-native prefill or decode
- **THEN** Runtime selects or builds a PreparedExecutionPlan whose guards match the current execution request.

#### Scenario: Missing plan fails closed
- **WHEN** no compatible PreparedExecutionPlan is available for first-native execution
- **THEN** Runtime rejects generation with a structured plan-unavailable or graph-planning error.

#### Scenario: Invalidated plan rejected
- **WHEN** a PreparedExecutionPlan has been invalidated
- **THEN** Runtime MUST NOT use it to produce logits for new work.

### Requirement: First-Native Evidence Identifies Plan Generation
First-native execution evidence SHALL identify the actual PreparedExecutionPlan generation used for each prefill and decode step.

#### Scenario: Plan generation observed
- **WHEN** a first-native execution step completes
- **THEN** observations include the PlanId or equivalent opaque identity and the PlanGeneration used by that step.

