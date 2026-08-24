# Define Kernel Registry And Dispatch

## Why

Magnetar now defines:

- Execution Graph
- Operator Contract
- Kernel Contract
- Provider
- Device
- Memory Manager
- Scheduler
- Resource Affinity
- Model Instance lifecycle
- Continuous Batching

The next missing layer is Kernel Registry and Dispatch.

The Kernel Contract defines what a Kernel is.

However, the Runtime also needs a mechanism to:

- register Kernel advertisements
- index Kernels by Operator
- validate Kernel candidates
- rank compatible candidates
- apply Runtime policy
- apply Provider and Device status
- apply Memory Manager feasibility
- preserve Resource Affinity
- perform explicit fallback
- create Kernel Invocations
- dispatch through the owning Provider
- collect structured results

Without this contract, kernel use may become hardcoded inside Providers,
Execution Graph planning, Scheduler, or model architecture code.

That would break portability and prevent Magnetar from supporting multiple
Providers and Devices cleanly.

This change defines Kernel Registry and Dispatch.

## What Changes

This change introduces:

- Kernel Registry
- Kernel Candidate
- Kernel Selection Request
- Kernel Selection Result
- Kernel Dispatch Plan
- Kernel Dispatch
- Kernel Fallback Chain
- Kernel dispatch lifecycle
- Kernel registry invalidation
- Provider/Device status integration
- Memory Manager integration
- Scheduler integration
- conformance gating
- observability

The exact Rust type names are implementation-defined.

## Kernel Registry

Kernel Registry SHALL be a Runtime-owned index of Kernel advertisements.

The registry SHALL index Kernels by:

- Operator identity
- Operator version compatibility
- Provider identity
- Device class
- dtype support
- layout support
- shape constraints
- memory class support
- execution mode
- Resource Affinity constraints
- conformance profile
- feature flags

Kernel Registry SHALL not execute Kernels.

It only tracks metadata and eligibility.

## Registry Ownership

Kernel Registry SHALL be owned by the Runtime.

Providers may advertise Kernels.

Runtime validates advertisements before accepting them.

Clients and Components SHALL NOT register Kernels directly.

A Kernel may only enter the registry through a trusted Provider registration or
Runtime test fixture path.

## Kernel Advertisement Validation

Before a Kernel advertisement enters the registry, Runtime SHALL validate:

- Provider identity
- Provider lifecycle
- Kernel identity
- implemented Operator identity
- Operator version compatibility
- metadata schema
- dtype metadata
- layout metadata
- shape constraints
- memory class constraints
- execution mode metadata
- workspace metadata
- cancellation metadata
- determinism metadata
- precision metadata
- required Provider features
- required Device features
- conformance status where required

Invalid advertisements SHALL be rejected.

## Registry Invalidation

Kernel Registry SHALL react to Provider and Device lifecycle changes.

Registry entries may become unavailable when:

- Provider unregisters
- Provider fails
- Provider drains
- Provider becomes not ready
- Provider pressure becomes saturated
- Device disappears
- Device resets
- Device becomes unavailable
- Device memory pressure exceeds policy
- Kernel conformance is revoked
- policy changes
- Runtime shuts down

Invalidation SHALL not leave stale dispatchable Kernels.

## Kernel Candidate

A Kernel Candidate is a registry entry being considered for a specific Operator
invocation.

Candidate metadata SHOULD include:

- Kernel identity
- Provider identity
- Device compatibility
- Operator compatibility
- dtype compatibility
- layout compatibility
- shape compatibility
- memory compatibility
- workspace feasibility
- Resource Affinity compatibility
- determinism compatibility
- precision compatibility
- Provider readiness
- Device readiness
- pressure score
- conformance status
- estimated cost
- fallback rank
- rejection reason if incompatible

## Kernel Selection Request

Kernel Selection Request SHALL be Runtime-created.

It SHOULD include:

- request ID
- Operator invocation reference
- graph plan reference
- Model Instance reference where relevant
- input/output resource references
- dtype requirements
- layout requirements
- shape requirements
- memory class requirements
- Resource Affinity requirements
- determinism requirements
- precision requirements
- execution mode preference
- batching metadata
- KV cache metadata
- adapter metadata
- deadline or timeout
- Runtime policy
- observability correlation

Clients and Components SHALL NOT create authoritative Kernel Selection Requests
that bypass Runtime validation.

## Selection Pipeline

Kernel selection SHALL follow a deterministic pipeline.

The pipeline SHOULD include:

```text
1. operator candidate lookup
2. Operator version filtering
3. Provider lifecycle filtering
4. Device lifecycle filtering
5. shape compatibility filtering
6. dtype compatibility filtering
7. layout compatibility filtering
8. memory class compatibility filtering
9. workspace feasibility
10. Resource Affinity validation
11. determinism and precision validation
12. batching compatibility validation
13. adapter/KV cache compatibility validation
14. conformance gating
15. policy ranking
16. fallback chain construction
17. selected candidate publication
```

Runtime MAY optimize implementation but observable decisions SHALL remain
explainable.

## Policy Ranking

Runtime policy SHALL rank compatible Kernel Candidates.

Ranking MAY consider:

- explicit policy preferences
- Provider readiness
- Provider pressure
- Device readiness
- Device pressure
- memory pressure
- expected latency
- expected throughput
- workspace cost
- data movement cost
- layout conversion cost
- dtype conversion cost
- determinism
- precision
- conformance profile
- power/thermal hints where available
- browser/native target
- previous failure history

Policy ranking SHALL not override hard Resource Affinity constraints.

## Resource Affinity

Kernel selection SHALL preserve Resource Affinity.

If input resources are bound to a Provider or Device, Runtime SHALL only select a
compatible Kernel unless explicit movement, conversion, or rebuild is planned.

Kernel Registry SHALL not silently select a Kernel that requires hidden data
movement.

## Memory Manager Integration

Kernel selection SHALL consult Memory Manager for workspace and output
feasibility.

Memory Manager SHALL evaluate:

- input residency
- output allocation
- workspace allocation
- memory class compatibility
- staging requirements
- temporary layout conversion
- temporary dtype conversion
- provider-owned memory accounting
- browser memory limits
- memory pressure
- pending allocation policy

If memory is unavailable, selection may choose fallback, queue, or reject
according to policy.

## Dispatch Plan

Kernel selection SHALL produce a Kernel Dispatch Plan.

A Dispatch Plan SHOULD include:

- selected Kernel identity
- owning Provider
- target Device metadata
- Operator invocation
- input resource bindings
- output resource bindings
- workspace reservation
- explicit movement/conversion steps
- execution mode
- cancellation support
- timeout/deadline
- fallback chain
- observability correlation
- cleanup behavior
- expected result metadata

The Dispatch Plan SHALL not expose raw Provider handles or native function
pointers.

## Dispatch

Kernel Dispatch SHALL submit a Runtime-created Kernel Invocation to the owning
Provider.

Runtime SHALL validate the Dispatch Plan immediately before dispatch.

Validation SHOULD re-check:

- Provider readiness
- Provider admission
- Provider pressure
- Device readiness
- Device availability
- memory reservation validity
- Resource Affinity
- cancellation state
- operation/session/model lifecycle
- policy

Dispatch SHALL fail closed if the selected Kernel is no longer eligible.

## Dispatch Lifecycle

Kernel dispatch lifecycle states SHOULD include:

```text
planned
ready
submitted
running
completed
failed
cancel-requested
cancelled
timed-out
fallback-pending
fallback-running
released
```

The exact serialized names are implementation-defined.

## Fallback Chain

Fallback SHALL be explicit.

A Fallback Chain MAY include:

- alternate Kernel on same Provider/Device
- alternate Kernel on same Provider different Device
- alternate Provider
- explicit dtype conversion
- explicit layout conversion
- explicit data movement
- host execution
- rejection

Fallback SHALL not silently violate policy.

Fallback SHALL preserve or explicitly transform Resource Affinity.

Fallback SHALL be observable.

## No Hidden Provider Selection

Kernel Registry and Dispatch SHALL not expose Provider selection authority to
Components or clients.

Provider and Device selection remain Runtime-owned.

Requests may carry policy preferences only when allowed.

Preferences SHALL not override Resource Affinity or Capability constraints.

## Scheduler Relationship

Scheduler may request dispatch for planned graph work.

Scheduler SHALL not select raw native function pointers.

Scheduler may use Kernel metadata for batching, deadlines, and backpressure, but
Runtime-owned Kernel Registry and Dispatch perform final validation and dispatch.

## Execution Graph Relationship

Execution Graph planning SHALL produce Operator invocations and Kernel
requirements.

Kernel Registry resolves those requirements to Kernel Candidates.

Execution Graphs SHALL not embed raw native Kernel pointers.

## Continuous Batching Relationship

Continuous Batching may dispatch batched Kernel Invocations.

Batch compatibility SHALL be validated against Kernel metadata.

Kernel Dispatch SHALL preserve per-operation output mapping.

A Kernel that cannot preserve batching semantics SHALL not be selected for that
batch.

## Adapter Relationship

Adapter-aware dispatch SHALL validate active adapter set against Kernel
metadata.

If adapter state changes after planning but before dispatch, Runtime SHALL
revalidate the dispatch plan.

## KV Cache Relationship

KV-cache-aware dispatch SHALL validate KV cache layout, dtype, memory class,
Resource Affinity, and lifecycle before dispatch.

If KV cache state becomes invalid after planning, dispatch SHALL fail or replan
according to policy.

## Prefix Cache Relationship

Prefix Cache affects dispatch indirectly through adjusted prefill boundaries,
sequence lengths, context lengths, and backing KV cache.

Dispatch SHALL use the validated graph plan and shall not bypass Prefix Cache
privacy or sharing policy.

## Provider Relationship

Providers advertise Kernels and execute Runtime-created Kernel Invocations.

Provider health, readiness, pressure, admission, and lifecycle SHALL influence
Kernel eligibility and dispatch.

Provider shall not execute unvalidated external Kernel requests.

## Device Relationship

Device metadata and state SHALL influence Kernel eligibility.

Device compatibility includes:

- device class
- memory classes
- dtype support
- layout support
- execution limits
- hardware feature flags
- readiness
- pressure
- availability

Device loss after selection but before dispatch SHALL cause revalidation failure
or fallback.

## Conformance Gating

Runtime policy MAY require Kernel conformance before dispatch.

Conformance gating MAY be based on:

- Operator family
- Provider type
- production mode
- safety level
- deterministic mode
- precision policy
- dynamic Provider trust
- test profile

Kernels failing required conformance SHALL not be selected.

## Dispatch Result

Kernel Dispatch SHALL return structured results.

Results SHOULD include:

- selected Kernel identity
- Provider identity
- Device metadata
- success or failure
- output readiness
- updated resource metadata
- timing metadata
- fallback used
- cancellation result
- determinism metadata
- precision diagnostics
- Provider diagnostics
- Device diagnostics
- structured error

Results SHALL not expose raw Provider handles, Device handles, memory pointers,
or raw tensor values.

## Error Model

Kernel Registry and Dispatch errors SHALL be structured.

Error categories SHOULD include:

- kernel registry unavailable
- kernel advertisement invalid
- kernel registration denied
- kernel candidate not found
- kernel candidate incompatible
- kernel selection failed
- kernel policy denied
- kernel conformance required
- kernel conformance missing
- kernel conformance failed
- kernel Provider unavailable
- kernel Provider not ready
- kernel Provider saturated
- kernel Device unavailable
- kernel Device incompatible
- kernel Device lost
- kernel memory infeasible
- kernel workspace unavailable
- kernel Resource Affinity conflict
- kernel dispatch plan invalid
- kernel dispatch stale
- kernel dispatch rejected
- kernel dispatch failed
- kernel fallback unavailable
- kernel fallback failed
- kernel cancellation unsupported
- kernel cancelled
- kernel timeout
- kernel browser feature unsupported
- internal kernel registry error
- internal kernel dispatch error

## Observability

Runtime SHOULD emit observations for:

- kernel advertisement received
- kernel advertisement accepted
- kernel advertisement rejected
- kernel registry updated
- kernel registry invalidated
- kernel candidate lookup
- kernel candidate rejected
- kernel candidate ranked
- kernel selected
- dispatch plan created
- dispatch plan revalidated
- dispatch submitted
- dispatch running
- dispatch completed
- dispatch failed
- fallback considered
- fallback selected
- fallback failed
- conformance gating applied
- memory feasibility failed
- Resource Affinity conflict
- Provider pressure affected selection
- Device pressure affected selection

Observability SHALL not expose raw tensor values, prompts, weights, KV cache
contents, Provider handles, Device handles, memory pointers, or function
pointers by default.

## Browser Target

Kernel Registry and Dispatch SHALL be platform-neutral.

Browser targets may support a reduced registry and dispatch model based on:

- browser-compatible Providers
- WebAssembly linear memory
- JavaScript-mediated execution
- future WebGPU buffers
- browser memory policy

Browser dispatch SHALL not require Wasmtime or native Provider loading.

Unsupported browser dispatch features SHALL return structured errors.

## Non-Goals

This change does not:

- define concrete CUDA kernels
- define concrete Metal kernels
- define concrete QNN kernels
- define concrete OpenVINO kernels
- define WebGPU kernel implementation
- define Provider kernel ABI
- define full graph optimizer
- define model architecture Components
- define distributed kernel dispatch
- define cross-node Provider selection
- define remote execution
- expose raw function pointers
- expose raw Provider handles to Components
- require GPU hardware
- require browser kernel implementation

## Impact

Magnetar gains the missing bridge between portable Operators and concrete
Provider Kernels.

The execution stack becomes:

```text
Execution Graph
    |
    v
Operator Invocation
    |
    v
Kernel Registry
    |
    v
Kernel Selection
    |
    v
Kernel Dispatch Plan
    |
    v
Provider Kernel Invocation
    |
    v
Kernel Result
```

This prepares:

- Model Component Contract
- first concrete operator catalog implementation
- first CPU Provider kernels
- later CUDA/Metal/OpenVINO/QNN kernels