# memory Specification

## Purpose
TBD - created by archiving change define-runtime-memory-manager. Update Purpose after archive.
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

