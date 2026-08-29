# memory Specification

## Purpose
This specification defines runtime memory planning, allocation classes, admission, accounting, dtype/layout relationships, and resource lifecycle guarantees.
## Requirements
### Requirement: Runtime Memory Manager

Magnetar SHALL provide a first-class Runtime Memory Manager.

The Memory Manager SHALL own Runtime memory allocation, residency, staging,
zero-copy feasibility, dtype placement, and memory pressure decisions.

#### Scenario: Allocate tensor memory

Given a Compute plan requires tensor memory

When Planning evaluates feasibility

Then Planning asks the Memory Manager

And does not allocate memory directly.

---

### Requirement: Memory Module Ownership

Memory Manager implementation SHALL be owned by a dedicated memory module or
equivalent first-class Runtime subsystem.

Memory allocation logic SHALL NOT be hidden inside Compute, Device, Provider,
or Scheduler modules.

#### Scenario: Inspect source ownership

Given a developer searches for allocator behavior

When they inspect the Runtime source tree

Then allocator behavior is owned by the Memory Manager subsystem.

---

### Requirement: Caching Allocator

The Memory Manager SHALL define a caching allocator model.

The caching allocator MAY retain released allocations for reuse according to
policy.

#### Scenario: Reuse allocation

Given a tensor allocation is released

And a compatible allocation request arrives later

When the caching allocator policy permits reuse

Then the Memory Manager may reuse the cached allocation.

---

### Requirement: Asynchronous Arena Allocation

The Memory Manager SHALL support asynchronous allocation semantics for
allocations that cannot complete immediately.

#### Scenario: Allocation pending

Given device memory is temporarily unavailable

And policy permits waiting

When an allocation request is submitted

Then the Memory Manager may place it in a pending allocation queue.

---

### Requirement: Pending Allocation Queues

Pending allocation queues SHALL be explicit Runtime state.

A pending request SHALL eventually complete, fail, time out, or be cancelled.

#### Scenario: Pending request times out

Given an allocation request is pending

And its deadline expires

When the Memory Manager evaluates the queue

Then the request fails with a structured allocation timeout error.

---

### Requirement: Memory Allocation Classes

The Memory Manager SHALL distinguish allocation classes.

Allocation classes SHOULD include tensor, model artifact, tokenizer artifact,
adapter artifact, quantization artifact, KV cache, prefix cache, temporary
workspace, transfer staging, pinned host memory, browser linear memory, and
Runtime internal memory.

#### Scenario: KV cache allocation

Given future generation requires KV cache memory

When allocation is requested

Then the request uses a KV cache allocation class

And is not treated as generic anonymous memory.

---

### Requirement: Memory Placement

The Memory Manager SHALL distinguish memory placement.

Placements MAY include ordinary host memory, pinned host memory, device memory,
unified/shared memory, provider-owned opaque memory, external borrowed memory,
browser linear memory, and staged temporary memory.

#### Scenario: Device placement requested

Given a tensor must reside on a Device

When the Memory Manager evaluates placement

Then it validates Device support and memory availability.

---

### Requirement: Pinned Host Memory

Pinned host memory SHALL be represented separately from ordinary host memory.

Pinned host memory SHALL be Runtime-managed and policy-limited.

#### Scenario: Host-device transfer

Given a transfer would benefit from pinned host memory

When policy and platform support it

Then the Memory Manager may allocate pinned host staging memory.

---

### Requirement: Components Cannot Directly Request Pinned Memory

Portable Components SHALL NOT directly request arbitrary pinned host memory.

Pinned memory may only be used through Runtime-mediated inference operations.

#### Scenario: Component requests pinned memory

Given a Component attempts to request pinned memory directly

When Runtime validates the request

Then the request is rejected or mapped through an authorized inference
operation.

---

### Requirement: Zero-Copy Feasibility

Zero-copy SHALL be represented as a Memory Manager feasibility decision.

Zero-copy SHALL NOT be assumed merely because two resources are visible to the
Runtime.

#### Scenario: Zero-copy unavailable

Given source and target placements are incompatible

When Memory Manager evaluates zero-copy

Then it returns zero-copy unavailable with a stable reason.

---

### Requirement: Host Staging Policy Enforcement

The Memory Manager SHALL enforce HostStagingPolicy.

If host staging is forbidden, Runtime SHALL NOT insert host staging silently.

#### Scenario: Staging forbidden

Given a data movement descriptor forbids host staging

When direct movement is impossible

Then Memory Manager rejects the plan instead of silently staging through host.

---

### Requirement: Storage DType And Compute DType

The Memory Manager SHALL distinguish storage dtype from compute dtype.

Storage dtype SHALL determine memory representation.

Compute dtype SHALL determine execution representation.

#### Scenario: Quantized storage with BF16 compute

Given model weights use INT8 storage

And execution uses BF16 compute

When Memory Manager plans memory

Then allocation size and temporary compute workspace are computed from both
storage and compute dtype.

---

### Requirement: Tensor Residency

The Memory Manager SHALL track tensor residency.

Tensor residency SHALL be Runtime-owned state and SHALL not be forgeable by
portable Components.

#### Scenario: Tensor resides on Provider

Given a tensor is materialized in Provider-owned memory

When dependent work is planned

Then Runtime uses Memory Manager residency and Resource Affinity

And does not trust Component-provided placement claims.

---

### Requirement: Model Residency Placeholder

The Memory Manager SHALL prepare for model residency ownership.

Model residency SHALL be distinct from Component Artifact identity and raw
filesystem paths.

#### Scenario: Future model loading

Given a future Model Artifact is loaded

When its weights are materialized

Then their residency is tracked by Memory Manager.

---

### Requirement: KV Cache Residency Placeholder

The Memory Manager SHALL prepare for KV cache memory ownership.

KV cache memory SHALL not be hidden inside Scheduler, Provider, or Generation
implementation.

#### Scenario: Future generation session

Given a future generation session creates KV cache

When memory is allocated

Then the allocation belongs to the KV cache allocation class.

---

### Requirement: Memory Pressure

The Memory Manager SHALL report memory pressure.

Memory pressure MAY be scoped by Runtime, Provider, Device, allocation class,
arena, cache, model residency, or KV cache.

#### Scenario: Device memory high pressure

Given Device memory pressure becomes high

When Scheduling evaluates new work

Then Runtime policy may delay, queue, or reject memory-heavy work.

---

### Requirement: Memory Admission

The Memory Manager SHALL provide memory admission decisions.

Admission decisions SHOULD include admit, queue, reject, and retry-later.

#### Scenario: Allocation rejected

Given memory pressure is saturated

And policy does not allow queueing

When a memory-heavy operation requests admission

Then Memory Manager rejects the request with a structured reason.

---

### Requirement: Browser Memory Placement

The Memory Manager SHALL represent browser memory constraints separately from
native host and device memory.

#### Scenario: Browser target

Given Magnetar is compiled for `wasm32`

When memory placement is evaluated

Then browser linear memory constraints are represented explicitly

And native pinned memory is not assumed.

---

### Requirement: Memory Errors

Memory Manager failures SHALL use structured error categories.

#### Scenario: Unsupported placement

Given an operation requires pinned memory

And the platform does not support pinned memory

When Memory Manager evaluates the request

Then it returns an unsupported placement or pinned-memory-unavailable error.

---

### Requirement: Memory Observability

The Memory Manager SHALL make structured observations available for allocation, release,
cache, arena, pending queue, staging, zero-copy, and pressure events.

#### Scenario: Cache hit

Given an allocation is served from the caching allocator

When Runtime observes the allocation

Then it may emit a memory cache hit observation.

### Requirement: Memory Manager Evaluates Model Artifact Feasibility

The Runtime Memory Manager SHALL evaluate Model Artifact loading feasibility
before model data becomes resident.

Feasibility SHALL consider storage dtype, compute dtype requirements,
quantization workspace, sharding metadata, adapter residency placeholders,
placement constraints, transfer staging, and memory pressure.

#### Scenario: Model loading exceeds memory policy

Given a Model Artifact requires more memory than current policy permits

When Runtime requests loading feasibility

Then the Memory Manager rejects the load with a structured feasibility failure.

---

### Requirement: Memory Manager Distinguishes Artifact Bytes From Residency

The Runtime Memory Manager SHALL distinguish persisted Model Artifact bytes from
resident model memory.

Resident model memory MAY differ from artifact bytes due to decompression,
dtype conversion, quantization workspace, sharded placement, provider-owned
buffers, or Runtime-owned staging.

#### Scenario: Quantized weights require workspace

Given Model Artifact bytes are stored in a quantized dtype

When Memory Manager plans residency

Then the plan accounts for compute-ready memory and dequantization workspace
separately from compressed storage bytes.

### Requirement: Memory Manager Supports Tokenizer Residency

Memory Manager SHALL support tokenizer artifact and vocabulary residency where
needed.

#### Scenario: Tokenizer vocabulary loaded

Given tokenizer vocabulary data is loaded

When Runtime records residency

Then Memory Manager may track vocabulary memory usage.

---

### Requirement: Memory Manager Supports Token Buffers

Memory Manager SHALL support tokenization-related buffers.

Token buffers MAY include encode output buffers, batch token buffers, attention
masks, token type IDs, and streaming decode state.

#### Scenario: Batch encoding memory

Given batch encoding requires token output buffers

When Memory Manager evaluates the request

Then it admits, queues, or rejects according to memory policy.

### Requirement: Memory Manager Supports Generation Admission

Memory Manager SHALL support generation memory admission.

Generation memory MAY include input token buffers, output token buffers, logits
buffers, sampling workspace, prefill workspace, decode workspace, and future KV
cache memory.

#### Scenario: Generation memory denied

Given generation requires more memory than policy permits

When Memory Manager evaluates admission

Then generation is rejected, queued, or delayed according to policy.

---

### Requirement: Memory Manager Prepares KV Cache Memory Boundary

Memory Manager SHALL prepare for KV cache memory requirements without requiring
full KV cache semantics in this change.

#### Scenario: Future KV cache estimate

Given generation estimates KV cache memory needs

When Memory Manager evaluates feasibility

Then it treats the memory as KV-cache-related allocation class or placeholder.

### Requirement: Memory Manager Tracks Session Memory

Memory Manager SHALL support session-scoped memory accounting.

#### Scenario: Session memory usage

Given a session allocates output token buffers

When Memory Manager reports usage

Then the allocation is associated with the session.

---

### Requirement: Memory Manager Enforces Session Budget

Memory Manager SHALL enforce session memory budgets according to Runtime policy.

#### Scenario: Session budget exceeded

Given a session has a memory budget

And a generation operation exceeds it

When memory admission is evaluated

Then Memory Manager rejects, queues, or delays according to policy.

---

### Requirement: Memory Manager Releases Session Resources

Memory Manager SHALL release session-scoped memory when the session closes,
expires, fails, or is cancelled according to policy.

#### Scenario: Close session

Given a session owns temporary workspace memory

When the session closes

Then Memory Manager releases the workspace.

### Requirement: Memory Manager Owns KV Cache Memory

Memory Manager SHALL allocate, track, admit, pressure-score, and release KV
cache memory.

#### Scenario: Allocate KV cache

Given generation requires KV cache memory

When Runtime plans generation

Then Memory Manager admits or rejects KV cache allocation.

---

### Requirement: Memory Manager Tracks KV Cache Residency

Memory Manager SHALL track KV cache residency.

Residency MAY include host memory, device memory, provider-owned memory,
browser linear memory, or future WebGPU buffers.

#### Scenario: Provider-owned residency

Given a Provider creates KV cache in native memory

When Runtime records the cache

Then Memory Manager tracks provider-owned residency metadata.

---

### Requirement: Memory Manager Handles KV Cache Pressure

Memory Manager SHALL include KV cache in memory pressure accounting.

#### Scenario: KV cache pressure high

Given KV cache memory usage is high

When Runtime evaluates memory pressure

Then KV cache pressure contributes to admission and eviction policy.

---

### Requirement: Memory Manager Releases Evicted Cache

When KV cache is evicted or released, Memory Manager SHALL release associated
resources according to ownership policy.

#### Scenario: Evict cache

Given a KV cache uses Device memory

When Runtime evicts the cache

Then Memory Manager releases or invalidates the Device memory record.

### Requirement: Memory Manager Owns Model Residency

Memory Manager SHALL track loaded model residency.

Model residency SHALL include placement, size, dtype, ownership, pressure, and
Resource Affinity metadata where applicable.

#### Scenario: Model loaded on Device

Given weights are materialized on Device memory

When residency is recorded

Then Memory Manager tracks Device residency and associated Resource Affinity.

---

### Requirement: Memory Manager Evaluates Loading Feasibility

Memory Manager SHALL evaluate model loading feasibility before materialization.

#### Scenario: Model too large

Given model loading requires more memory than policy permits

When feasibility is evaluated

Then Memory Manager rejects, queues, or delays loading according to policy.

---

### Requirement: Memory Manager Releases Model Residency

Memory Manager SHALL release model residency memory when unload policy requires
it.

#### Scenario: Unload releases memory

Given a loaded model owns Device memory

When Runtime unloads the model

Then Memory Manager releases associated memory records.

---

### Requirement: Memory Manager Accounts For Quantization Transform Workspace

Memory Manager SHALL account for temporary workspace required by quantization,
dequantization, or Provider-specific model transforms.

#### Scenario: Dequantization workspace

Given INT8 weights must be converted to BF16 during loading

When loading is planned

Then Memory Manager accounts for temporary BF16 workspace.

---

### Requirement: Memory Manager Supports Pending Model Loading

Memory Manager SHALL support policy-controlled queuing for model loading allocations.

#### Scenario: Loading queued

Given memory pressure prevents immediate allocation

And policy permits waiting

When model loading requests memory

Then Memory Manager may place the loading request in a pending allocation queue.

### Requirement: Memory Manager Supports Sampling Buffers

Memory Manager SHALL account for Sampling temporary buffers.

Buffers MAY include logits, probabilities, masks, sorted token workspace, top-k
workspace, top-p workspace, RNG state, history buffers, and penalty workspace.

#### Scenario: Sampling workspace denied

Given top-p Sampling requires workspace memory

When Memory Manager denies allocation

Then Sampling fails with memory-allocation-failed or queues according to policy.

---

### Requirement: Memory Manager Controls Logits Materialization

Memory Manager SHALL participate in logits materialization decisions.

#### Scenario: Host logits materialization

Given Device-resident logits must be materialized on host

When Memory Manager policy denies staging

Then Sampling fails or chooses another compatible path.

### Requirement: Memory Manager Accounts For Prefix Cache

Memory Manager SHALL account for Prefix Cache metadata, index memory, lookup
workspace, and backing KV cache references.

#### Scenario: Prefix metadata allocation

Given Prefix Cache creates an entry

When metadata is allocated

Then Memory Manager accounts for that memory.

---

### Requirement: Memory Pressure May Evict Prefix Cache

Runtime SHALL allow Memory Manager pressure to trigger Prefix Cache eviction according to Runtime
policy.

#### Scenario: Memory pressure

Given prefix cache memory pressure is high

When Runtime applies eviction policy

Then Prefix Cache entries may be evicted.

### Requirement: Memory Manager Supports Batch Admission

Memory Manager SHALL support memory admission for continuous batching.

Batching memory MAY include input buffers, output buffers, logits buffers,
attention masks, position buffers, sampling workspace, KV cache blocks, Prefix
Cache lookup workspace, temporary staging, and Provider-specific workspace.

#### Scenario: Batch memory denied

Given a planned batch exceeds memory policy

When Memory Manager evaluates admission

Then Runtime reduces, queues, or rejects the batch according to policy.

---

### Requirement: Memory Pressure Influences Batch Size

Memory pressure SHALL influence batch sizing and admission.

#### Scenario: High memory pressure

Given memory pressure is high

When Scheduler forms the next batch

Then it may reduce batch size or delay prefill according to policy.

### Requirement: Memory Manager Tracks Adapter Residency

Memory Manager SHALL track adapter residency separately from base model
residency.

#### Scenario: Adapter resident

Given adapter tensors are loaded into Device memory

When Runtime reports memory usage

Then adapter residency is accounted separately.

---

### Requirement: Memory Manager Enforces Adapter Memory Budget

Memory Manager SHALL enforce adapter memory budgets according to Runtime and
session policy.

#### Scenario: Adapter budget exceeded

Given a session adapter memory budget is exceeded

When adapter loading is requested

Then Memory Manager rejects, queues, or delays according to policy.

---

### Requirement: Memory Manager Accounts For Adapter Merge Workspace

Memory Manager SHALL account for workspace required to merge, unmerge, or
transform adapters.

#### Scenario: Merge workspace

Given merge-on-activation requires temporary workspace

When Runtime plans activation

Then Memory Manager accounts for merge workspace.

### Requirement: Memory Manager Tracks Model Instance Residency

Memory Manager SHALL track all residency associated with Model Instances.

#### Scenario: Instance residency report

Given a Model Instance has host and device residency

When memory usage is queried

Then Memory Manager reports residency by instance metadata.

---

### Requirement: Memory Manager Coordinates Instance Lifecycle

Memory Manager SHALL coordinate allocation, residency update, suspension,
unload, eviction, and release with Model Instance lifecycle.

#### Scenario: Instance unload

Given a Model Instance is unloading

When Runtime releases residency

Then Memory Manager releases or invalidates all associated memory records.

---

### Requirement: Memory Pressure May Affect Instance Readiness

Runtime SHALL define how memory pressure may cause a Model Instance to become suspended, draining,
unloaded, or failed according to Runtime policy.

#### Scenario: Memory pressure

Given Runtime memory pressure is high

When policy permits instance suspension

Then Runtime may mark idle instance suspended and release eligible memory.

### Requirement: Memory Manager Plans Graph Memory

Memory Manager SHALL participate in Execution Graph memory planning.

Graph memory MAY include tensor edges, operator outputs, workspace, layout
conversions, dtype conversions, KV cache inputs/outputs, adapter paths, and
temporary buffers.

#### Scenario: Operator workspace

Given an operator requires workspace

When Runtime plans the graph

Then Memory Manager admits, queues, or rejects workspace allocation.

---

### Requirement: Memory Manager Tracks Tensor Edge Residency

Memory Manager SHALL track residency for graph tensor edges where those edges
correspond to Runtime-managed allocations.

#### Scenario: Tensor output

Given an operator writes tensor T to Device memory

When execution completes

Then Memory Manager tracks T residency and Resource Affinity.

---

### Requirement: Memory Manager Prevents Silent Movement

Memory Manager SHALL require explicit Runtime-planned data movement, dtype
conversion, or layout conversion.

#### Scenario: Host staging forbidden

Given graph planning would require host staging

And policy forbids host staging

When planning runs

Then planning fails instead of silently staging.

### Requirement: Memory Manager Allocates Kernel Workspace

Kernel workspace SHALL be allocated through Memory Manager.

#### Scenario: Kernel workspace

Given Kernel dispatch requires temporary workspace

When Runtime plans dispatch

Then Memory Manager admits, queues, or rejects the workspace allocation.

---

### Requirement: Memory Manager Validates Kernel Memory Classes

Memory Manager SHALL validate that input, output, and workspace memory classes
are compatible with Kernel requirements.

#### Scenario: Pinned host required

Given Kernel requires pinned host memory

When Memory Manager cannot provide it

Then Runtime rejects the Kernel or chooses fallback.

---

### Requirement: Memory Manager Tracks Kernel Resource Effects

Memory Manager SHALL track resource metadata changes caused by Kernel execution.

#### Scenario: Kernel writes output

Given Kernel writes an output tensor

When execution completes

Then Memory Manager records output readiness, residency, and Resource Affinity.

### Requirement: Memory Manager Participates In Kernel Selection

Memory Manager SHALL participate in Kernel candidate feasibility checks for
inputs, outputs, workspace, staging, movement, dtype conversion, and layout
conversion.

#### Scenario: Workspace infeasible

Given a Kernel requires workspace larger than allowed

When selection runs

Then Memory Manager rejects the candidate.

---

### Requirement: Memory Reservations Are Revalidated Before Dispatch

Memory reservations required by a Dispatch Plan SHALL be revalidated before
Kernel dispatch.

#### Scenario: Reservation expired

Given workspace reservation expires before dispatch

When revalidation runs

Then dispatch fails stale or replans according to policy.

---

### Requirement: Dispatch Results Update Memory Metadata

Kernel Dispatch results SHALL update Memory Manager metadata for output
readiness, residency, Resource Affinity, workspace release, and provider-owned
memory accounting.

#### Scenario: Kernel output ready

Given Kernel writes output tensor

When dispatch completes

Then Memory Manager records output readiness and residency metadata.

---

### Requirement: Host Memory Supports Reference CPU Kernels

Memory Manager SHALL support host memory resources usable by Reference CPU
Kernels.

#### Scenario: CPU input tensor

Given input tensor is host-resident

When CPU Kernel dispatch runs

Then Memory Manager provides Runtime resource references for host access.

---

### Requirement: CPU Outputs Are Tracked

Outputs produced by Reference CPU Kernels SHALL be tracked by Memory Manager.

#### Scenario: CPU output

Given CPU matmul writes output tensor

When dispatch completes

Then Memory Manager marks output ready and host-resident.

---

### Requirement: CPU Fallback Requires Explicit Movement

If fallback to Reference CPU requires moving data to host memory, movement SHALL
be explicit and policy-controlled.

#### Scenario: Device tensor fallback

Given tensor is Device-resident

When CPU fallback is considered

Then Runtime plans explicit movement or rejects fallback.

### Requirement: First Scope Memory Is Host-Compatible

The first operator implementation scope SHALL be executable with host memory
through Reference CPU Provider.

#### Scenario: Host tensor

Given required-now operator input is host-resident

When CPU Kernel dispatch runs

Then Memory Manager tracks input/output host residency.

---

### Requirement: Unsupported Layout Movement Is Explicit

Unsupported layout movement SHALL be explicit when first scope requires layout
conversion through Memory Manager and graph planning.

#### Scenario: Non-contiguous input

Given input layout is unsupported

When first scope planning runs

Then Runtime inserts explicit conversion where available or rejects the graph.

### Requirement: Memory Manager Owns Tensor Resource Allocation

Memory Manager SHALL allocate, track, admit, release, evict, and invalidate Tensor Resources.

#### Scenario: Allocate tensor

Given Runtime plans output tensor

When Memory Manager admits allocation

Then Tensor Resource metadata is created or updated.

---

### Requirement: Memory Manager Tracks Tensor Residency

Memory Manager SHALL track Tensor Resource residency, memory class, host visibility, Provider/Device affinity, transfer state, conversion state, and eviction eligibility.

#### Scenario: Tensor transfer

Given Tensor moves from Device to host

When transfer completes

Then Memory Manager updates residency and Resource Affinity metadata.

---

### Requirement: Memory Manager Validates Tensor Size

Memory Manager SHALL compute or conservatively estimate Tensor Resource size.

#### Scenario: Unknown packed size

Given packed quantized tensor size cannot be computed

When admission runs

Then Memory Manager rejects or applies conservative policy.

---

### Requirement: Memory Manager Tracks Tensor Views

Memory Manager SHALL track Tensor View lifetime dependency on base resources.

#### Scenario: Base tensor evicted

Given a view depends on base tensor

When base tensor is evicted

Then the view becomes invalid or unavailable.

---

### Requirement: Memory Manager Enforces Tensor Mutability And Aliasing

Memory Manager SHALL participate in mutability and aliasing validation where storage ownership is affected.

#### Scenario: Immutable tensor mutation

Given Tensor Resource is immutable

When Kernel requests write access

Then Memory Manager rejects or reports mutability violation.

---

### Requirement: Memory Baseline Precedes Provider Execution

Memory Manager baseline SHALL be available before Reference CPU Provider writes
Runtime-visible outputs.

#### Scenario: CPU output allocation

Given CPU matmul dispatches

When output is required

Then Memory Manager tracks allocation and readiness.

---

### Requirement: Memory Baseline Tracks Cleanup

Memory baseline SHALL support cleanup sufficient for E2E conformance.

#### Scenario: Session close cleanup

Given E2E session closes

When cleanup runs

Then inference-scoped resources are released or retained only according to
policy.

---

### Requirement: Post-Baseline Memory Classes Are Tracked

Device, pinned-host, unified, shared, provider-owned, browser-linear-memory, and WebGPU buffer memory classes SHALL be tracked by Memory Manager when supported.

#### Scenario: CUDA device output

Given CUDA Kernel writes Device output

When dispatch completes

Then Memory Manager tracks memory class, residency, and Resource Affinity.

---

### Requirement: Provider Data Movement Is Explicit

Post-baseline Provider data movement SHALL be explicit and policy-controlled.

#### Scenario: Device to host fallback

Given fallback to CPU requires Device-to-host transfer

When planning runs

Then Runtime inserts explicit movement or rejects fallback.

### Requirement: Model Formats Feed Memory Planning

Normalized model format metadata SHALL provide enough size, dtype, layout, and
shard metadata for Memory Manager planning.

#### Scenario: Sharded model

Given sharded artifact metadata is normalized

When Memory Manager plans loading

Then it can estimate or compute required memory.

---

### Requirement: Memory Mapping Is Policy-Controlled

If memory mapping is supported for model formats, it SHALL be policy-controlled
and SHALL not expose raw mmap pointers through public APIs.

#### Scenario: mmap requested

Given safetensors loading uses memory mapping

When Runtime reports metadata

Then no raw memory pointer is exposed.

### Requirement: Cache Storage Is Not Memory Residency

Artifact cache storage SHALL be distinct from Runtime memory residency.

#### Scenario: Cached artifact

Given artifact exists in cache

When Runtime memory is inspected

Then artifact tensors are not resident unless loaded through Memory Manager.

---

### Requirement: Model Loading Materializes From Cache Through Memory Manager

When loading from cache, Model Loading SHALL still materialize Tensor Resources
through Memory Manager.

#### Scenario: Load cached weights

Given cached weights are valid

When Model Loading materializes weights

Then Memory Manager tracks resulting Tensor Resources.

### Requirement: Memory Release Gate

Memory Manager baseline SHALL have release gate coverage.

#### Scenario: Untracked allocation

Given Runtime-visible allocation is not tracked

When release validation runs

Then stable release is blocked.

---

### Requirement: Cache Is Not Memory Residency Gate

Release gates SHALL validate cache storage is distinct from memory residency.

#### Scenario: Cached model

Given model is cached but not loaded

When Memory Manager is inspected

Then model tensors are not resident.

---

### Requirement: Kernel Executable Memory Is Distinct From Tensor Memory

Provider-owned executable kernel memory SHALL remain distinct from Runtime
Tensor Resource memory.

#### Scenario: CUDA module loaded

Given Provider allocates executable device memory

When tensor residency is inspected

Then executable allocation is not treated as model tensor allocation.

---

### Requirement: Kernel Preparation Does Not Transfer Tensor Ownership

Kernel preparation SHALL NOT transfer Runtime Tensor Resource ownership to
Provider.

#### Scenario: Prepared kernel references buffers at invocation

Given kernel executes using Runtime tensors

When invocation completes

Then tensor lifecycle remains controlled by Memory Manager.

---

### Requirement: Compilation Workspace Is Distinct From Inference Tensor Memory

Compiler temporary/workspace memory SHALL not be confused with Runtime Tensor
Resource residency.

#### Scenario: Compiler uses 2 GiB host memory

Given compilation is active

When inference tensor accounting is inspected

Then compiler workspace is classified separately.

---

### Requirement: Compilation Resource Pressure May Affect Admission

Runtime SHALL account for compilation resource pressure when deciding whether to start additional cold-path work.

#### Scenario: Host pressure high

Given compilation would exceed configured resource policy

When job is submitted

Then Runtime queues or rejects cold-path compilation.

---

### Requirement: Compilation Never Owns Runtime Tensor Resources

Compiler SHALL NOT obtain ownership of inference Tensor Resources as part of
normal compilation.

#### Scenario: Shape specialization

Given compiler needs tensor shapes

When request is built

Then metadata is provided rather than mutable inference tensor ownership.

---

### Requirement: Kernel Cache Is Not Tensor Residency

Persistent Kernel Artifact cache SHALL remain distinct from Runtime Tensor
residency.

#### Scenario: Cached CUBIN

Given CUBIN is stored on disk

When Memory Manager reports model tensor residency

Then CUBIN cache storage is not counted as resident inference tensor.

---

### Requirement: Prepared Kernel Executable Memory Is Distinct

Prepared Kernel executable memory SHALL remain logically distinct from Runtime
Tensor Resource memory.

#### Scenario: GPU module loaded

Given Provider allocates module memory

When tensor memory accounting runs

Then executable module memory is not treated as model tensor ownership.

---

### Requirement: Kernel Retirement Does Not Free Runtime Tensor Memory

Destroying Prepared Kernel SHALL NOT implicitly free Runtime-owned tensor
allocations.

#### Scenario: Hot swap

Given old kernel is retired

When Provider destroys native kernel state

Then model weights/KV/tensor resources remain governed by Memory Manager.

---

### Requirement: Memory Manager Is Authoritative For Feasibility

Kernel selection SHALL respect Memory Manager feasibility decisions.

#### Scenario: Fast Kernel workspace exceeds capacity

Given Kernel A is faster

But its workspace cannot be admitted

When selection runs

Then Kernel A is excluded.

---

### Requirement: Memory Cost May Influence Ranking

Runtime SHALL NOT use memory cost to override Memory Manager feasibility decisions, though among feasible candidates memory/workspace cost MAY influence optimization.

#### Scenario: Memory profile

Given two feasible Kernels

When memory profile is active

Then lower memory candidate may rank higher.

---

### Requirement: Selection Does Not Allocate Hidden Memory

Kernel ranking SHALL NOT perform hidden inference allocations.

#### Scenario: Candidate evaluated

Given candidate requires workspace

When eligibility is evaluated

Then feasibility is checked without silently committing unmanaged allocation.

---

### Requirement: Memory Manager Governs Autotuning Workspace

Autotuning candidate SHALL not benchmark with memory resources unavailable to
normal production policy.

#### Scenario: Fast variant needs excessive workspace

Given workspace exceeds allowed memory

When tuning plans candidate

Then candidate is rejected or skipped as production-infeasible.

---

### Requirement: Tuning Allocations Are Temporary

Autotuning-specific tensor/workspace allocations SHALL be released after tuning
lifecycle.

#### Scenario: Benchmark finishes

Given candidate benchmark used temporary buffers

When candidate measurement completes

Then buffers are reclaimed according to Runtime policy.

---

### Requirement: Tuning Cannot Steal Unbounded Inference Memory

Runtime SHALL bound memory consumed by tuning.

#### Scenario: Active model under pressure

Given tuning budget would violate reserved inference capacity

When admission occurs

Then tuning is denied/postponed.
