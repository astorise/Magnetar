# kernel-registry Specification

## Purpose
This specification defines runtime-owned kernel registry selection, dispatch planning, Provider readiness checks, and fail-closed kernel compatibility.
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

---

### Requirement: Reference CPU Kernels Enter Registry Through Validation

Reference CPU Kernel advertisements SHALL be validated before registry
insertion.

#### Scenario: Invalid CPU advertisement

Given Reference CPU Provider advertises unknown Operator

When Runtime validates it

Then the advertisement is rejected.

---

### Requirement: Reference CPU Candidate Selection

Reference CPU Kernels SHALL participate in normal Kernel candidate lookup,
filtering, ranking, fallback, and dispatch.

#### Scenario: CPU candidate

Given graph contains matmul

When Kernel Registry queries candidates

Then Reference CPU matmul may be considered if advertised and policy allows.

---

### Requirement: Reference CPU Fallback Observable

Fallback to Reference CPU SHALL be explicit and observable.

#### Scenario: CPU fallback used

Given optimized Kernel is unavailable

And policy permits CPU fallback

When Runtime selects Reference CPU

Then observability records fallback usage.

### Requirement: Registry Supports First Scope Validation

Kernel Registry SHALL support validation that required-now operators have
eligible Kernels.

#### Scenario: Validate first scope

Given first scope requires RMSNorm

When Kernel Registry is checked

Then at least one eligible RMSNorm Kernel must exist or validation fails.

---

### Requirement: Registry Does Not Create Placeholder Candidates

Kernel Registry SHALL not create candidates for placeholder Operators unless a
Provider advertises a concrete Kernel.

#### Scenario: Placeholder lookup

Given no Provider advertises paged-attention

When Registry is queried

Then no candidate is returned.

---

### Requirement: Registry Reports Missing Required Kernels

Kernel Registry SHALL report missing required-now Kernels with structured
errors.

#### Scenario: Missing attention kernel

Given attention is required-now

And no eligible Kernel exists

When first scope validation runs

Then Runtime reports first-scope-kernel-missing.

---

### Requirement: E2E Uses Kernel Registry

E2E conformance SHALL validate Kernel Registry candidate lookup and selection
for required operators.

#### Scenario: Matmul kernel selected

Given graph contains matmul

When execution is planned

Then Kernel Registry selects an eligible Reference CPU matmul Kernel.

---

### Requirement: E2E Detects Missing Kernels

E2E conformance SHALL include missing kernel failure cases.

#### Scenario: Missing attention kernel

Given attention Operator has no eligible Kernel

When E2E graph execution is planned

Then Runtime reports structured missing Kernel error.

---

### Requirement: Kernel Registry Precedes E2E Execution

Kernel Registry and Dispatch SHALL be implemented before E2E local inference
success path.

#### Scenario: E2E matmul

Given E2E graph contains matmul

When execution runs

Then Kernel Registry selects an eligible Reference CPU Kernel.

---

### Requirement: Kernel Dispatch Revalidation Included In Baseline

Kernel Dispatch baseline SHALL revalidate Provider, Device, Memory, Resource
Affinity, and policy before dispatch.

#### Scenario: Provider unavailable

Given selected Provider becomes unavailable

When dispatch begins

Then dispatch fails closed or replans according to policy.

---

### Requirement: Registry Handles Optimized Provider Candidates

Kernel Registry SHALL support optimized Provider candidates without bypassing
normal validation and ranking.

#### Scenario: CUDA and CPU candidates

Given CUDA and Reference CPU kernels exist for matmul

When Kernel Registry ranks candidates

Then it validates compatibility, readiness, memory, policy, and Resource
Affinity before selection.

---

### Requirement: Registry Requires Conformance Metadata For Advanced Features

Kernel Registry SHALL NOT select an advanced-feature Kernel candidate that lacks required conformance metadata.
Kernel Registry SHOULD consider conformance metadata for advanced Provider
features.

#### Scenario: Unconformant flash attention

Given flash attention Kernel lacks required conformance

When Registry selects candidates

Then the Kernel is rejected or ranked unavailable according to policy.

---

### Requirement: Registry Tracks Prepared Kernel Readiness

A Kernel candidate without an associated ready PreparedKernel SHALL NOT be
treated as immediately dispatchable. Kernel Registry MAY associate compatible
Kernel candidates with PreparedKernel state.

#### Scenario: Kernel not prepared

Given compatible artifact exists but no PreparedKernel is ready

When dispatch selection runs

Then candidate is not treated as immediately executable.

---

### Requirement: Registry Does Not Own Native Handles

Kernel Registry SHALL NOT store or dereference native executable pointers.

#### Scenario: Prepared CUDA Kernel

Given Provider owns native CUDA function

When Registry stores candidate

Then it stores opaque PreparedKernelId only.

---

### Requirement: Registry Supports Multiple Prepared Generations

An older Prepared Kernel generation SHALL remain valid for in-flight requests
until no active reference remains. Kernel Registry MAY temporarily index
multiple Prepared Kernel generations for the same logical Kernel.

#### Scenario: Hot replacement

Given generation 18 replaces 17

When new request is dispatched

Then policy may choose 18 while in-flight request continues using 17.

---

### Requirement: Registry Validates Artifact Compatibility

Kernel Registry SHALL use artifact metadata as part of compatibility
selection where applicable.

#### Scenario: Architecture mismatch

Given compiled artifact targets incompatible architecture

When candidate selection occurs

Then candidate is excluded.

---

### Requirement: Registry Does Not Compile

Kernel Registry SHALL NOT perform source compilation.

#### Scenario: Missing compiled artifact

Given only source artifact exists

When Registry selects candidates

Then it reports preparation unavailable rather than invoking a compiler itself.

---

### Requirement: Registry Considers Qualification Eligibility

Kernel Registry SHALL consider qualification status when policy requires it.

#### Scenario: Faster unqualified Kernel

Given unqualified candidate benchmarks faster

When production selection runs

Then it cannot outrank eligible qualified candidates.

---

### Requirement: Registry Promotion Is Explicit

Candidate Kernel SHALL become active only through explicit promotion.

#### Scenario: Candidate prepared

Given candidate is prepared successfully

When no promotion occurs

Then it does not automatically become active.

---

### Requirement: Atomic Kernel Promotion

Dispatch SHALL NOT observe a partially updated Registry state.

Registry promotion SHOULD be atomic from dispatch perspective.

#### Scenario: Promotion races with dispatch

Given promotion occurs concurrently with new invocation

When dispatch resolves Kernel

Then invocation observes complete old or complete new Registry generation.

---

### Requirement: Multiple Prepared Generations

Each tracked generation SHALL be uniquely identified.

Registry MAY track multiple Prepared Kernel generations.

#### Scenario: Hot swap

Given generation 2 is promoted

When generation 1 has in-flight work

Then both generations may coexist temporarily.

---

### Requirement: Retiring Generation Receives No New Work

After Kernel generation enters retiring state, Registry SHALL stop selecting it
for new work.

#### Scenario: New request after promotion

Given old Kernel is retiring

When request resolves

Then new active generation is selected where compatible.

---

### Requirement: Revoked Kernel Not Selected

Registry SHALL not select revoked Kernel for new work.

#### Scenario: Security revocation

Given active Kernel is revoked

When next dispatch occurs

Then revoked Kernel is not selected.

---

### Requirement: Performance Ranking Follows Eligibility

Performance ranking SHALL occur after compatibility, qualification, trust and
policy eligibility.

#### Scenario: Incorrect fastest candidate

Given incorrect candidate has best benchmark

When Registry ranks

Then candidate remains ineligible.

---

### Requirement: Registry Provides Candidate Set

Kernel Registry SHALL expose candidate metadata to Runtime selection policy
without performing opaque policy ranking itself.

#### Scenario: Multiple MatMul kernels

Given Registry contains four compatible MatMul implementations

When selection begins

Then Runtime policy can evaluate candidates explicitly.

---

### Requirement: Registry Eligibility Metadata

Registry candidate metadata SHALL include information required for eligibility
evaluation.

#### Scenario: Qualified candidate

Given Registry returns candidate

When Runtime evaluates it

Then qualification, trust, target and preparation state are available.

---

### Requirement: Registry Does Not Make Cross-Provider Optimization Decision

Kernel Registry SHALL not independently choose the globally fastest Provider
without Runtime selection policy.

#### Scenario: CPU and CUDA candidates

Given both exist

When Kernel is selected

Then Runtime policy decides according to eligibility and objective.

---

### Requirement: Registry Supports Stable Candidate Identity

Candidate identity SHALL be stable enough for deterministic tie-breaking.

#### Scenario: Equal benchmark scores

Given candidates tie

When stable key is compared

Then deterministic ordering is available.

---

### Requirement: Registry Respects Revocation

Revoked candidate SHALL be excluded before optimization ranking.

#### Scenario: Previously fastest Kernel revoked

Given fastest candidate becomes revoked

When Registry candidates are evaluated

Then it cannot be selected.

### Requirement: Registry Discovers Only Accepted Candidates

Normal Kernel Registry candidate discovery SHALL exclude staged, quarantined,
rejected, and revoked ingestion artifacts.

#### Scenario: Candidate still validating

Given ingestion transaction has not committed

When Registry discovers Operator implementations

Then candidate is absent.

---

### Requirement: Registry Publication Follows Commit

Candidate metadata SHALL NOT become Registry-discoverable until ingestion
commit succeeds and any additional required registration policy is satisfied.

#### Scenario: Commit succeeds

Given accepted Kernel is committed

When candidate registration runs

Then Registry may index it according to normal policy.

---

### Requirement: Ingestion Failure Does Not Mutate Registry

Failed import SHALL not partially create Registry candidate.

#### Scenario: Qualification evidence malformed

Given manifest parsing succeeded but evidence validation failed

When transaction rejects

Then Registry state is unchanged.

---

### Requirement: Registry Can Distinguish Specialization Instances

Kernel Registry SHALL retain enough identity to distinguish Runtime-relevant
specializations.

#### Scenario: Two Attention variants

Given two prepared variants have different tile specialization

When candidates are evaluated

Then each may carry distinct performance evidence.

---

### Requirement: Registry Does Not Tune Automatically

Kernel Registry SHALL not start benchmarks or compilation as a side effect of
candidate lookup.

#### Scenario: Candidate lookup

Given tuning record is absent

When Registry returns candidates

Then no autotuning session is implicitly started.

---

### Requirement: Registry Selection Uses Normal Policy After Tuning

Autotuning evidence MAY inform ranking but SHALL not bypass Registry/Runtime
eligibility.

#### Scenario: Winner revoked

Given tuning record names fastest candidate

But Registry marks it revoked

Then it is excluded.

---

### Requirement: Registry Preserves Performance Evidence Identity

Registry SHALL associate performance evidence with the correct Kernel Artifact,
specialization, and generation context.

#### Scenario: New Kernel generation

Given generation N+1 replaces N

When performance evidence is queried

Then N observations do not silently become N+1 observations.

---

### Requirement: Registry Does Not Generate Performance Evidence

Kernel Registry SHALL not fabricate missing benchmark or online metrics.

#### Scenario: No online samples

Given candidate lacks observations

When ranking occurs

Then evidence remains missing rather than inferred from another candidate.

### Requirement: Registry Resolution May Feed Plan Construction

Kernel Registry SHALL expose candidate metadata needed to construct Prepared
Execution Plan.

#### Scenario: Plan build

Given three eligible MatMul candidates

When Plan construction runs

Then Runtime selection policy resolves one candidate and records exact binding.

---

### Requirement: Ready Plan Avoids Repeated Full Registry Resolution

Execution of compatible ready Plan SHALL not require complete Registry
candidate discovery on every invocation.

#### Scenario: Repeated decode

Given same compatible Plan remains ready

When successive token steps execute

Then node bindings can reuse Plan decisions.

---

### Requirement: Registry Change Does Not Mutate Existing Plan

Kernel Registry preference change SHALL not rewrite an already acquired Plan
generation.

#### Scenario: Kernel v3 promoted

Given active Plan references v2

When Registry changes preference to v3

Then existing Plan becomes stale/replacement candidate rather than changing
binding in place.

---

### Requirement: Revocation Propagates To Dependent Plans

Registry/revocation system SHALL permit Runtime to identify Plans requiring a
revoked Kernel.

#### Scenario: Security revocation

Given Kernel digest is revoked

When Runtime evaluates dependent Plans

Then they become invalid for new work.

### Requirement: First-Native Qwen Operators Resolve Through Registry
Every executable operator in the first-native Qwen graph SHALL resolve through Kernel Registry before Provider dispatch.

#### Scenario: Operator dispatch has registry lineage
- **WHEN** Runtime executes a Qwen graph node
- **THEN** evidence links GraphNodeId to KernelRegistryResolutionId, KernelId, PreparedKernelId, ProviderSubmissionId, and CompletionId.

#### Scenario: Required kernel disabled
- **WHEN** a required Qwen kernel is unavailable or disabled
- **THEN** plan preparation or execution fails rather than calling a direct Reference CPU bypass.

### Requirement: Reference CPU Direct Calls Are Not Model E2E Execution
Direct Reference CPU kernel functions SHALL NOT execute the authoritative first-native Qwen model E2E path.

#### Scenario: Direct calls remain unit-test-only
- **WHEN** Reference CPU functions are used in unit tests, qualification oracles, or differential tests
- **THEN** they are allowed only outside the authoritative model E2E execution path.

