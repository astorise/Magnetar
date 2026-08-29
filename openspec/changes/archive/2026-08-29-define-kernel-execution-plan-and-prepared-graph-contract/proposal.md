# Define Kernel Execution Plan And Prepared Graph Contract

## Why

Magnetar now defines the complete decision chain required to execute
hardware-specialized Kernels safely:

```text
Execution Graph
    -> Kernel discovery
    -> eligibility
    -> qualification/trust checks
    -> optimization policy
    -> specialization/autotuning
    -> memory feasibility
    -> Provider preparation
    -> Kernel selection
```

These operations belong primarily to cold or warm Runtime paths.

Repeating the complete resolution process for every token, Operator invocation,
or continuous-batching step would unnecessarily increase hot-path overhead.

Magnetar therefore needs a first-class prepared execution representation that
captures already validated Runtime decisions while remaining safely
invalidatable when their assumptions change.

This change introduces the Prepared Execution Plan.

## What Changes

This change defines:

- Execution Graph fingerprint
- PreparedExecutionPlan
- PreparedExecutionPlanId
- Plan Generation
- Plan Node Binding
- Prepared Execution Segment
- Resource Binding Plan
- Plan Guards
- workload compatibility
- prefill/decode plan separation
- dynamic shape guards
- plan lifecycle
- plan readiness
- soft staleness
- hard invalidation
- invalidation reasons
- re-planning
- atomic plan replacement
- in-flight plan generation safety
- Provider-prepared graph/segment state
- plan cache
- plan identity
- Memory Manager integration
- continuous batching integration
- Model Instance integration
- adaptive feedback integration
- observability
- conformance

## Core Principle

Portable semantic graphs and prepared execution decisions SHALL remain
distinct.

```text
ExecutionGraph
    =
portable computation semantics

PreparedExecutionPlan
    =
Runtime-selected execution strategy for a compatible context
```

A Prepared Execution Plan SHALL NOT redefine Operator semantics.

## Canonical Flow

The canonical preparation flow is:

```text
Model Component
      |
      v
Execution Graph
      |
      v
Graph Validation
      |
      v
Kernel Candidate Resolution
      |
      v
Eligibility Filtering
      |
      v
Kernel Selection
      |
      v
Specialization Resolution
      |
      v
Memory Planning
      |
      v
Provider / Kernel Preparation
      |
      v
Prepared Execution Plan
      |
      v
READY
```

Execution then becomes:

```text
Prepared Execution Plan
      |
      v
guard validation
      |
      v
resource binding
      |
      v
prepared dispatch
```

## Execution Graph Remains Authoritative Semantics

The original Execution Graph SHALL remain the semantic source of truth.

Prepared Execution Plan SHALL reference the graph identity/fingerprint from
which it was derived.

A Plan SHALL NOT silently add, remove, reorder, fuse, or reinterpret semantic
operations unless such transformation is already allowed by explicit
Operator/fusion semantics and validated by Runtime.

## Execution Graph Fingerprint

Runtime SHALL be able to identify the semantic graph from which a Plan was
derived.

A graph fingerprint SHOULD include or derive from:

- Operator IDs
- Operator semantic versions
- portable attributes
- graph topology
- tensor logical descriptors
- explicit conversion/movement nodes
- fused semantic groups where applicable

It SHALL exclude process-local native handles.

## Prepared Execution Plan

A PreparedExecutionPlan represents an immutable or generation-stable Runtime
decision describing how one compatible Execution Graph should execute.

Conceptually:

```text
PreparedExecutionPlan
    id
    generation
    graph_fingerprint
    model_instance_revision
    scope
    guards
    node_bindings
    segments
    resource_plan
    policy_fingerprints
    state
```

## Plan Identity

PreparedExecutionPlan SHALL have stable Runtime identity.

Plan identity SHOULD incorporate or reference enough information to distinguish
material execution decisions, including:

```text
Execution Graph fingerprint
Model Instance revision
Kernel Artifact digests
KernelIds
Specialization Instance IDs
Provider bindings
Device compatibility
selection policy version
memory-plan version
relevant autotuning evidence
```

The identity SHALL NOT contain native pointers.

## Plan Generation

A logical execution scope MAY have multiple plan generations.

Example:

```text
plan generation 12
    active

plan generation 13
    preparing

plan generation 14
    candidate
```

Promotion from one plan generation to another SHALL be atomic from the
perspective of new executions.

## Plan Scope

A Plan SHALL declare the workload/context scope for which it is valid.

Scope MAY include:

- Model Instance
- execution phase
- workload bucket
- shape envelope
- dtype/layout
- Provider/Device compatibility
- quantization mode
- batching mode
- KV-cache mode
- adapter configuration revision
- policy revision

## Model Instance Binding

Prepared Execution Plan SHALL be bound to the Model Instance revision for which
it was created.

Changes that alter execution semantics or required resources MAY invalidate the
Plan.

Examples include:

```text
model replacement
adapter set change
merged adapter revision
execution policy change
graph revision
```

## Plan Node Binding

Each executable graph node or validated fused group MAY have a Plan Node
Binding.

Conceptually:

```text
PlanNodeBinding
    graph node(s)
    KernelId
    Kernel Artifact digest
    Specialization Instance
    ProviderBinding
    DeviceBinding
    PreparedKernelId
    execution mode
    resource bindings
```

PreparedKernelId SHALL remain opaque.

## Kernel Binding

A Plan binding SHALL identify the exact Kernel implementation and
specialization chosen for that Plan generation.

It SHOULD identify:

- KernelId
- artifact digest
- qualification profile
- specialization
- Prepared Kernel generation
- Provider
- Device compatibility

where relevant.

## No Native Handles

Prepared Execution Plan SHALL NOT expose or interpret:

- function pointers
- CUDA function pointers
- Metal objects
- Vulkan handles
- command-buffer pointers
- streams
- queues
- contexts
- Provider-native graph objects

Any native executable state remains Provider-private.

## PreparedKernelId

PreparedKernelId MAY appear in internal Runtime Plan state as an opaque
Provider-owned reference.

It SHALL NOT be serialized as portable Plan identity.

A persisted/reconstructed Plan SHALL require Provider re-preparation unless a
Provider-specific safe mechanism explicitly recreates the prepared state.

## Prepared Execution Segment

Runtime MAY group multiple compatible graph nodes into a Prepared Execution
Segment.

A segment MAY represent:

- fused Kernel group
- Provider graph capture
- Provider command sequence
- execution subgraph
- batched submission group

A segment SHALL preserve graph semantics.

## Segment Ownership

Runtime owns the logical segment definition.

Provider MAY own native prepared segment state.

Conceptually:

```text
PreparedExecutionSegment
    logical nodes
    ProviderBinding
    DeviceBinding
    opaque ProviderPreparedSegmentId
```

## ProviderPreparedSegmentId

Provider MAY return an opaque identifier for prepared/captured graph state.

This identifier SHALL follow the same rules as PreparedKernelId:

- opaque
- Provider-owned
- not a native pointer
- not portable
- not exposed through WIT
- not exposed through Runtime Inference API

## Cross-Provider Segmentation

One Prepared Execution Plan MAY contain multiple Provider/Device segments only
when Runtime policy explicitly permits the associated Resource Affinity and
data movement.

Cross-Provider movement SHALL remain explicit.

A Plan SHALL NOT hide transfers required between segments.

## Plan Resource Bindings

Prepared Execution Plan MAY precompute logical resource binding requirements.

These MAY include:

- input slots
- output slots
- model-weight resources
- temporary workspace slots
- KV-cache slots
- adapter resources
- intermediate tensor slots
- explicit data-movement resources

## Resource Binding Plan

ResourceBindingPlan SHALL describe resource requirements and relationships.

It SHALL NOT transfer Memory Manager ownership to the Plan.

The Memory Manager remains authoritative for:

- allocation
- residency
- lifetime
- eviction
- movement feasibility
- affinity

## Stable Versus Dynamic Resources

Plan SHALL distinguish stable resources from invocation/session resources.

Examples of stable resources:

```text
model weights
immutable adapter weights
persistent Provider-prepared constants
```

Examples of dynamic resources:

```text
input token tensors
per-request outputs
session KV cache
continuous-batch slots
temporary workspace
```

A Plan SHALL not incorrectly capture one Session's resources as globally
reusable state.

## Resource Slots

Dynamic resource bindings SHOULD use logical slots rather than process-native
addresses.

Example:

```text
input:hidden-state
session:kv-key
session:kv-value
workspace:attention
output:logits
```

Exact representation is internal.

## KV Cache Boundary

Prepared Execution Plan MAY describe required KV-cache layout, affinity, and
resource slots.

It SHALL NOT own Session KV-cache contents.

Session/KV Cache lifecycle remains governed by existing Runtime contracts.

## Memory Plan

A Prepared Execution Plan MAY reference a Memory Plan.

The Memory Plan MAY precompute:

- reusable allocation classes
- lifetimes
- workspace upper bounds
- aliasing opportunities
- buffer reuse
- placement constraints

It SHALL remain subordinate to current Memory Manager feasibility.

## Memory Revalidation

Before executing a Plan, Runtime MAY perform lightweight memory guard
validation.

A Plan whose required resources are no longer feasible MAY be invalidated or
replanned according to policy.

## Dynamic Shapes

A Prepared Execution Plan MAY support dynamic shapes only within an explicit
validated envelope.

A Plan SHALL define guards for relevant dynamic dimensions.

Example:

```text
batch        = 1..8
sequence     = 1..4096
head_dim     = 128
```

An invocation outside the envelope SHALL not execute the Plan.

## Plan Guard

A Plan Guard is a cheap Runtime-checkable condition required for safe Plan use.

Plan guards MAY include:

- shape range
- dtype
- layout
- execution phase
- batch range
- sequence range
- adapter revision
- KV layout
- Provider readiness generation
- Device availability class
- resource-affinity compatibility
- policy compatibility

## Guard Cost

Plan guards SHALL be designed for bounded hot-path cost.

Expensive qualification, benchmarking, compilation, or broad candidate
resolution SHALL NOT occur during guard checking.

## Guard Failure

A guard failure SHALL result in:

```text
use compatible alternate Plan
request re-plan
fallback according to policy
structured failure
```

It SHALL NOT silently execute an incompatible Plan.

## Plan Families

Runtime MAY maintain a family of Plans for one Execution Graph.

Example:

```text
Attention plan family:

prefill / batch 1..8
prefill / batch 9..32
decode / seq <= 2048
decode / seq 2049..8192
```

Plan-family selection SHOULD use cheap workload guards.

## Prefill And Decode Plans

Prefill and decode MAY have distinct Prepared Execution Plans.

This permits different:

- Kernels
- specializations
- execution segments
- memory workspaces
- Provider graph captures

without changing model semantics.

## Continuous Batching

Prepared Execution Plan MAY support continuous batching within declared
constraints.

Its guards MAY include:

- maximum active sequences
- total token count
- raggedness class
- paged KV-cache compatibility
- batch-slot model

A Plan SHALL NOT assume one fixed request identity.

## Plan Build

Plan construction SHALL occur outside normal Kernel execution hot path.

Plan build MAY perform:

- graph analysis
- Registry queries
- qualification checks
- selection
- autotuning-record lookup
- memory planning
- Kernel preparation
- Provider segment preparation

## Plan Build Is Not Arbitrary Optimization

Plan construction SHALL consume already authorized Kernels and policies.

It SHALL NOT invoke arbitrary AI generation.

It SHALL NOT start an Optimization Campaign synchronously.

## Plan Build And Autotuning

Plan construction MAY consume existing Autotuning Records.

Policy MAY allow bounded warmup autotuning before Plan becomes ready.

Such tuning SHALL follow the Runtime Autotuning contract.

## Plan Build And Compilation

If a required accepted Kernel specialization needs compilation, Plan
preparation MAY request Provider compilation during cold/warm path according to
policy.

Compilation SHALL not occur during execution of a ready Plan.

## Plan State

Suggested Plan states are:

```text
building
validating
preparing
ready
stale
invalidated
retiring
retired
failed
```

## Ready

`ready` means all mandatory Plan bindings required by the declared scope are
prepared and current hard invariants are satisfied.

## Stale

`stale` means the Plan remains safe to execute temporarily but policy considers
its optimization evidence or preference potentially outdated.

Examples:

- newer Kernel promoted
- tuning record stale
- performance model requests re-evaluation
- selection policy preference changed without invalidating correctness

A stale Plan MAY continue executing according to policy while replacement is
built.

## Invalidated

`invalidated` means the Plan is no longer safe or policy-eligible for new
execution.

Examples:

- Kernel revoked
- qualification revoked where required
- Provider unavailable
- Device unavailable
- required artifact revoked
- Resource Affinity invalid
- Model Instance revision changed incompatibly
- Prepared Kernel destroyed
- hard memory assumption invalid
- security policy denies current binding

Invalidated Plan SHALL receive no new work.

## Stale Versus Invalidated

The distinction SHALL be explicit.

```text
stale
    =
suboptimal or evidence outdated,
but still safe/eligible according to current policy

invalidated
    =
must not accept new execution
```

Performance regression alone SHOULD normally produce stale/demotion/replan
behavior unless it also exposes a contract or safety violation.

## Invalidation Reasons

Structured invalidation reasons SHOULD include:

```text
graph-changed
model-instance-revision-changed
adapter-revision-changed
kernel-revoked
qualification-revoked
trust-denied
kernel-generation-unavailable
prepared-kernel-destroyed
provider-unavailable
device-unavailable
device-health-invalid
resource-affinity-invalid
memory-plan-invalid
policy-invalid
hard-compatibility-change
```

## Staleness Reasons

Structured staleness reasons SHOULD include:

```text
kernel-promotion
better-candidate-available
selection-policy-updated
autotuning-record-stale
performance-regression
workload-drift
benchmark-drift
device-pressure-profile-changed
memory-pressure-profile-changed
```

## Plan Revalidation

Runtime SHOULD support lightweight Plan revalidation.

Revalidation SHALL not repeat all expensive Plan construction work when the
existing Plan remains provably compatible.

## Replanning

A stale or invalid Plan MAY produce a PlanRebuildRequest.

Replanning SHALL occur outside the active execution hot path.

## No Hot-Path Full Replanning

Normal token decode SHALL NOT synchronously:

- discover all Kernel candidates
- rerun qualification
- benchmark candidates
- compile Kernels
- rebuild complete Memory Plan
- recreate Provider graph captures

as a consequence of Plan staleness.

## Safe Boundary Switching

Runtime SHOULD switch Plan generations only at a safe execution boundary.

Possible boundaries include:

- before model invocation
- between prefill and decode
- between decode steps
- between continuous-batch scheduling quanta
- between graph segments where semantics permit

Runtime SHALL not replace Provider-native execution state underneath an
in-flight invocation.

## Plan Generation Lease

An active invocation SHALL acquire a logical lease/reference to the Plan
generation it uses.

A retiring Plan SHALL not be destroyed until active leases reach zero or
equivalent quiescence is established.

## Atomic Plan Replacement

Replacement Plan publication SHALL be atomic for new executions.

Conceptually:

```text
generation 20 active
generation 21 ready

atomic publish

generation 20 retiring
generation 21 active
```

New executions see complete generation 20 or generation 21, never a mixture of
binding metadata.

## Partial Node Replacement

Runtime MAY eventually support partial Plan replacement.

If implemented, it SHALL preserve an atomic coherent Plan view.

A node binding SHALL not be changed independently in a way that creates an
unvalidated mixed execution strategy.

A new Plan generation SHOULD be preferred for coherent replacement.

## Kernel Hot Swap Integration

Kernel promotion may make an existing Plan stale.

The Plan SHALL not automatically start using a new Prepared Kernel simply
because Kernel Registry preferred generation changed.

A new Plan generation SHALL capture the new binding.

This preserves deterministic in-flight behavior.

## Kernel Revocation Integration

Kernel revocation SHALL invalidate Plans that require that Kernel for new work.

Runtime MAY switch to an already-ready fallback Plan if available.

Otherwise it SHALL request re-planning or fail according to policy.

## Performance Feedback Integration

Adaptive Performance Model MAY mark a Plan stale or request re-planning.

It SHALL NOT mutate the Plan in place.

## Selection Policy Integration

Selection policy version/fingerprint SHOULD be associated with the Plan.

A materially changed policy MAY mark a Plan stale or invalid depending on
policy severity.

Security constraints SHALL take precedence over performance preference.

## Autotuning Integration

A Plan SHOULD record the Autotuning Record or specialization evidence used for
its bindings where relevant.

If that evidence becomes stale:

```text
Plan may become stale
```

rather than immediately invalid if the Kernel remains safe and eligible.

## Plan Cache

Runtime MAY cache Prepared Execution Plan metadata.

The Plan cache SHALL be distinct from:

- Kernel Artifact Cache
- Kernel Autotuning Cache
- Model Artifact Cache
- Prefix Cache
- KV Cache

## Plan Cache Key

A Plan cache key SHOULD include compatibility-relevant context such as:

```text
graph fingerprint
Model Instance revision
execution phase
workload bucket
Kernel Artifact digests
specialization IDs
Provider version
Device compatibility
selection-policy version
memory-policy version
adapter revision
KV layout
```

## Plan Metadata Persistence

Portable/persistent Plan metadata MAY record logical decisions.

Process-local prepared identifiers SHALL NOT be treated as reusable after
Runtime restart unless reconstructed safely.

A persisted Plan SHOULD therefore be considered:

```text
plan recipe / logical decision
```

until Provider preparation is re-established.

## Plan Cache Hit

A cached Plan SHALL still revalidate current hard dependencies before becoming
ready.

A cache hit SHALL not bypass:

- revocation
- trust
- qualification
- Provider readiness
- Device readiness
- Prepared Kernel reconstruction
- memory feasibility

## Plan Cache Invalidation

Relevant compatibility changes SHALL invalidate or stale cached Plan metadata.

## Provider Prepared Graph

A Provider MAY support preparing an execution segment as a native graph or
command structure.

Examples MAY include:

- CUDA Graph
- Metal command/pipeline structure
- Vulkan command graph/pipeline group
- OpenVINO compiled subgraph
- QNN graph
- WebGPU prepared pipeline sequence

Provider SHALL advertise such capability.

## Provider Prepared Graph Is Optional

Providers that only execute individual Kernels remain valid.

Prepared Execution Plan SHALL not require Provider-native graph capture.

## Provider Graph Ownership

Native Provider-prepared graph state SHALL remain Provider-owned.

Runtime receives only an opaque identifier and metadata.

## Provider Graph Compatibility

A Provider-prepared segment SHALL be bound to compatible:

- Provider
- Device
- Kernel generations
- resource-binding model
- dynamic-shape envelope

Changing incompatible dependencies SHALL invalidate the prepared segment.

## Graph Capture And Dynamic Buffers

Provider graph capture MUST NOT require Runtime to expose raw buffer pointers as
portable Plan identity.

Provider may internally patch/bind native addresses according to its own
contract at execution time.

## Graph Capture Failure

Failure to prepare a Provider graph segment MAY:

- fall back to individual prepared Kernel dispatch
- fail Plan preparation

according to explicit policy.

It SHALL NOT silently change semantics.

## Plan Execution

Execution of a ready Plan SHOULD avoid expensive candidate resolution.

Hot-path execution MAY consist of:

```text
lookup Plan
check guards
bind resources
acquire Plan generation lease
dispatch segment/kernel bindings
record completion
release lease
```

## Hot Path Objective

The objective is not "zero overhead".

The objective is:

```text
Runtime decision overhead
    << Provider/device execution cost
```

with bounded predictable operations.

## Error Model

Structured errors SHOULD include:

```text
kernel-execution-plan-not-found
kernel-execution-plan-not-ready
kernel-execution-plan-build-failed
kernel-execution-plan-validation-failed
kernel-execution-plan-preparation-failed
kernel-execution-plan-guard-failed
kernel-execution-plan-workload-incompatible
kernel-execution-plan-shape-incompatible
kernel-execution-plan-dtype-incompatible
kernel-execution-plan-layout-incompatible
kernel-execution-plan-phase-incompatible
kernel-execution-plan-model-revision-mismatch
kernel-execution-plan-adapter-revision-mismatch
kernel-execution-plan-kv-layout-incompatible

kernel-execution-plan-stale
kernel-execution-plan-invalidated
kernel-execution-plan-kernel-revoked
kernel-execution-plan-qualification-revoked
kernel-execution-plan-provider-unavailable
kernel-execution-plan-device-unavailable
kernel-execution-plan-affinity-invalid
kernel-execution-plan-memory-invalid
kernel-execution-plan-prepared-kernel-missing

kernel-execution-plan-rebuild-required
kernel-execution-plan-rebuild-failed
kernel-execution-plan-replacement-failed
kernel-execution-plan-generation-in-use
kernel-execution-plan-retirement-failed

kernel-execution-segment-preparation-failed
kernel-execution-segment-invalid
kernel-execution-segment-provider-incompatible

kernel-execution-plan-hot-path-rebuild-denied
internal-kernel-execution-plan-error
```

## Observability

Plan observability MAY include:

```text
plan-build-started
plan-node-bound
plan-segment-created
plan-memory-planned
plan-preparation-started
plan-ready
plan-cache-hit
plan-cache-miss
plan-guard-failed
plan-marked-stale
plan-invalidated
plan-rebuild-requested
plan-replacement-ready
plan-generation-promoted
plan-retiring
plan-retired
```

Observability MAY expose:

- Plan ID
- Plan generation
- graph fingerprint
- Model Instance ID/revision
- Kernel IDs
- artifact digests
- specialization IDs
- Provider/Device stable bindings
- workload scope
- guard failure reason
- stale/invalidation reason
- policy version

Observability SHALL NOT expose:

- native Kernel handles
- Provider graph pointers
- raw tensor addresses
- raw model weights
- KV contents
- prompts
- secrets
- credentials

## Conformance

Conformance SHALL validate:

- Execution Graph remains semantic source of truth
- Plan does not change portable semantics
- Plan uses exact validated Kernel bindings
- native handles remain Provider-private
- dynamic resource slots do not capture Session-specific native addresses
- shape guards reject incompatible workload
- stale and invalidated states are distinct
- stale Plan may remain safe according to policy
- invalidated Plan receives no new work
- Kernel revocation invalidates dependent Plans
- Kernel promotion creates replacement Plan rather than mutating in-flight Plan
- Memory Manager remains authoritative
- Plan build happens outside normal decode hot path
- execution uses bounded guard/binding path
- atomic Plan replacement preserves coherent generation
- in-flight Plan generation remains alive until quiescent
- Plan cache does not bypass current eligibility
- Provider graph capture remains optional
- Provider-prepared segment state remains opaque
- adaptive feedback requests re-plan rather than mutating Plan
- reproducible/pinned Model Instance can retain Plan selection
- observability remains redacted

## Non-Goals

This change does not:

- create a new portable graph language
- replace Execution Graph
- expose native graph handles
- require CUDA Graph
- require Provider graph capture
- define distributed Plan scheduling
- serialize live Tensor Resources
- serialize KV-cache contents
- persist native PreparedKernelId across restart
- perform arbitrary AI optimization
- perform benchmarking in token decode
- make Plan immutable forever
- eliminate all Runtime dispatch checks

## Impact

Magnetar gains a prepared execution layer:

```text
portable semantic graph
        |
        v
expensive Runtime reasoning
        |
        v
Prepared Execution Plan
        |
        v
bounded hot-path guards/binding
        |
        v
Provider execution
```

The Runtime can therefore retain rich adaptive selection and safety policy
without paying the complete decision cost on every token or Operator
invocation.