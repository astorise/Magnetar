# kernel-registry Specification

## Purpose
TBD - created by archiving change define-kernel-registry-and-dispatch. Update Purpose after archive.
## Requirements
### Requirement: Kernel Registry

Magnetar SHALL define Kernel Registry as a Runtime-owned index of validated
Kernel advertisements.

#### Scenario: Provider advertises Kernel

Given a Provider advertises a matmul Kernel

When Runtime validates the advertisement

Then the Kernel may be inserted into the Kernel Registry.

---

### Requirement: Registry Is Runtime-Owned

Kernel Registry SHALL be owned by Runtime.

Clients and Components SHALL NOT register Kernels directly.

#### Scenario: Component registers Kernel

Given a Component attempts to register a native Kernel

When Runtime validates the request

Then registration is denied.

---

### Requirement: Registry Indexes Kernel Metadata

Kernel Registry SHALL index Kernel metadata by Operator, Provider, Device class,
dtype, layout, shape constraints, memory classes, execution mode, Resource
Affinity constraints, conformance profile, and feature flags.

#### Scenario: Lookup attention Kernels

Given graph planning needs attention

When Runtime queries the registry

Then candidate attention Kernels are found by Operator metadata.

---

### Requirement: Advertisement Validation

Runtime SHALL validate Kernel advertisements before registry insertion.

#### Scenario: Invalid advertisement

Given a Kernel advertisement references an unknown Operator

When Runtime validates it

Then the advertisement is rejected.

---

### Requirement: Registry Invalidation

Kernel Registry SHALL invalidate entries when Provider, Device, conformance, or
policy state makes them unavailable.

#### Scenario: Provider fails

Given a Provider fails

When Runtime updates registry state

Then Kernels owned by that Provider are no longer dispatchable.

---

### Requirement: Kernel Candidate

Kernel Registry SHALL produce Kernel Candidates for specific Operator
invocations.

Candidates SHALL include compatibility metadata and rejection reasons where
applicable.

#### Scenario: Candidate rejected

Given a Kernel supports FP16 only

When invocation requires BF16

Then the candidate is rejected with dtype incompatibility metadata.

---

### Requirement: Kernel Selection Request

Kernel selection SHALL use a Runtime-created selection request containing
validated Operator invocation, graph plan, resource, dtype, layout, shape,
memory, Resource Affinity, determinism, precision, batching, KV cache, adapter,
deadline, policy, and observability metadata.

#### Scenario: Selection request

Given graph planning produces an Operator invocation

When Kernel selection runs

Then Runtime creates the selection request.

---

### Requirement: Selection Pipeline

Kernel selection SHALL apply compatibility, lifecycle, memory, Resource
Affinity, conformance, and policy filters before selecting a Kernel.

#### Scenario: Provider not ready

Given a candidate Kernel belongs to a Provider that is not ready

When selection runs

Then the candidate is not selected.

---

### Requirement: Policy Ranking

Runtime policy SHALL rank compatible Kernel Candidates.

Hard Resource Affinity constraints SHALL not be overridden by ranking.

#### Scenario: Faster incompatible Kernel

Given Kernel A is faster but violates Resource Affinity

And Kernel B is compatible

When ranking runs

Then Kernel A is not selected.

---

### Requirement: Memory Feasibility

Kernel selection SHALL consult Memory Manager for output and workspace
feasibility.

#### Scenario: Workspace unavailable

Given a candidate Kernel requires workspace that cannot be allocated

When selection runs

Then the candidate is rejected or fallback is considered.

---

### Requirement: Dispatch Plan

Kernel selection SHALL produce a Kernel Dispatch Plan before dispatch.

The plan SHALL include selected Kernel, Provider, Device, resources, workspace,
explicit movement or conversion steps, execution mode, cancellation, fallback,
observability, cleanup, and expected result metadata.

#### Scenario: Dispatch plan

Given a Kernel candidate is selected

When selection completes

Then Runtime produces a Dispatch Plan.

---

### Requirement: Dispatch Revalidation

Kernel Dispatch SHALL revalidate Provider, Device, memory, Resource Affinity,
lifecycle, cancellation, and policy state immediately before dispatch.

#### Scenario: Device lost before dispatch

Given a Kernel was selected for Device A

But Device A is lost before dispatch

When revalidation runs

Then dispatch fails closed or fallback is attempted.

---

### Requirement: Dispatch Lifecycle

Kernel Dispatch SHALL expose lifecycle state.

States SHOULD include planned, ready, submitted, running, completed, failed,
cancel-requested, cancelled, timed-out, fallback-pending, fallback-running, and
released.

#### Scenario: Dispatch completed

Given Provider reports successful Kernel execution

When Runtime handles the result

Then dispatch lifecycle becomes completed and resources are updated.

---

### Requirement: Fallback Chain

Kernel fallback SHALL be explicit and policy-controlled.

Fallback SHALL not silently violate Resource Affinity, dtype, layout, memory,
determinism, precision, or Provider policy.

#### Scenario: Fallback conversion

Given no Kernel supports current layout

And policy allows layout conversion

When fallback is considered

Then Runtime plans explicit layout conversion before alternate Kernel dispatch.

---

### Requirement: No Hidden Provider Selection

Kernel Registry and Dispatch SHALL not grant Provider or Device selection
authority to clients or Components.

#### Scenario: Client preference

Given a client requests Provider `cuda`

When Kernel selection runs

Then the request is treated only as policy input if allowed and remains
non-authoritative.

---

### Requirement: Scheduler Does Not Select Raw Functions

Scheduler SHALL not select raw native Kernel function pointers.

Scheduler may request Runtime dispatch using validated metadata.

#### Scenario: Scheduler dispatches work

Given Scheduler schedules graph work

When Kernel execution is needed

Then Runtime Kernel Dispatch performs final selection and validation.

---

### Requirement: Graphs Do Not Embed Kernel Pointers

Execution Graphs SHALL not embed raw native Kernel function pointers.

#### Scenario: Graph metadata inspected

Given a graph is inspected

When metadata is returned

Then no native Kernel pointer is exposed.

---

### Requirement: Batched Dispatch Compatibility

Batched dispatch SHALL validate Kernel batch metadata and preserve
per-operation output mapping.

#### Scenario: Ragged batch unsupported

Given batch has ragged sequences

And selected Kernel does not support ragged batches

When dispatch is validated

Then dispatch is rejected or alternate Kernel is selected.

---

### Requirement: Adapter Revalidation

Kernel Dispatch SHALL revalidate active adapter compatibility before dispatch.

#### Scenario: Adapter changed after planning

Given Kernel plan was built for adapter A

When adapter B becomes active before dispatch

Then dispatch fails stale or replans according to policy.

---

### Requirement: KV Cache Revalidation

Kernel Dispatch SHALL revalidate KV cache lifecycle, layout, dtype, memory class,
and Resource Affinity before dispatch.

#### Scenario: KV cache invalidated

Given selected Kernel consumes KV cache

When the cache becomes invalid before dispatch

Then dispatch fails or replans according to policy.

---

### Requirement: Conformance Gating

Runtime policy SHALL support requiring Kernel conformance before selection or dispatch.

#### Scenario: Missing conformance

Given production policy requires conformance

And candidate Kernel lacks passing conformance

When selection runs

Then the candidate is rejected.

---

### Requirement: Dispatch Result

Kernel Dispatch SHALL return structured results without exposing raw handles,
function pointers, memory pointers, or raw tensor values.

#### Scenario: Dispatch success

Given Kernel execution succeeds

When Runtime returns dispatch result

Then output readiness and stable metadata are returned.

---

### Requirement: Registry And Dispatch Error Categories

Kernel Registry and Dispatch failures SHALL use structured error categories.

#### Scenario: No candidate

Given no compatible Kernel exists

When selection runs

Then Runtime returns kernel-candidate-not-found or kernel-selection-failed.

---

### Requirement: Registry And Dispatch Observability

Runtime SHALL support emitting observations for advertisements, registry updates,
candidate lookup, candidate ranking, selection, dispatch planning, dispatch,
fallback, conformance gating, pressure effects, and failures.

Observability SHALL not expose raw tensor values, prompts, weights, KV cache
contents, Provider handles, Device handles, memory pointers, or function
pointers by default.

#### Scenario: Kernel selected

Given a Kernel is selected

When observability records it

Then Runtime emits redacted selection metadata.

---

### Requirement: Browser-Compatible Registry And Dispatch

Kernel Registry and Dispatch SHALL be platform-neutral and SHALL not require
Wasmtime or native Provider loading.

#### Scenario: Browser target

Given Runtime runs on browser target

When a native-only Kernel is requested

Then Runtime returns kernel-browser-feature-unsupported or selects a
browser-compatible path.

