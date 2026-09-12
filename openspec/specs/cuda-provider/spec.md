# cuda-provider Specification

## Purpose
Defines the CUDA Provider's baseline contract: identity, graceful unavailability without compatible hardware, device discovery, Kernel advertisements and their correctness against Reference CPU, explicit data movement and Memory Manager integration (including caller-owned output pre-admission and Device-resident chaining between Kernels), synchronous execution, error categories, and conformance scope -- including native multi-head RoPE rotation.
## Requirements
### Requirement: CUDA RoPE Kernel Supports Native Multi-Head Rotation

The CUDA Provider's `rope` Kernel SHALL accept an optional `head_count`
parameter. When `head_count` is present and greater than `1`, the Kernel
SHALL divide each row into `head_count` equal-width blocks of
`head_width = cols / head_count` columns and, within each block
independently, rotate exactly the first `dimension` columns of that block
(`dimension` MAY be less than `head_width`, partial rotation) using the
same per-pair rotation formula already defined for the single-block case;
`position` is per-row, not per-head. Columns outside a rotated range SHALL
be copied through unchanged, never zeroed. `head_count` absent (or `1`)
SHALL reproduce the existing single-block rotation behavior exactly, with
`dimension` unconstrained by `head_count`'s divisibility rule.

Multi-head rotation SHALL execute as a single Kernel invocation regardless
of `head_count`; the Provider SHALL NOT require one invocation per head.

#### Scenario: Single-block call is unaffected

- **GIVEN** a `rope` Kernel invocation with no `head_count` attribute (or
  `head_count = 1`)
- **WHEN** the CUDA Provider dispatches it
- **THEN** the rotated output is bit-for-bit identical to this Provider's
  pre-existing single-block `rope` behavior

#### Scenario: Multi-head call with different Q and K head counts

- **GIVEN** two `rope` Kernel invocations over the same row count, one with
  `head_count` equal to a model's attention head count and one with
  `head_count` equal to a smaller grouped-query key/value head count
- **WHEN** the CUDA Provider dispatches each
- **THEN** each rotates its own row width using its own `head_width = cols
  / head_count`, independently of the other invocation's head count

#### Scenario: Partial RoPE within a head is supported

- **GIVEN** a `rope` Kernel invocation with `head_count > 1` and a rotation
  `dimension` strictly less than `head_width = cols / head_count`
- **WHEN** the CUDA Provider dispatches it
- **THEN** only the first `dimension` columns of each head's block are
  rotated
- **AND** the remaining columns of that block are copied from input to
  output unchanged, not zeroed

#### Scenario: Multi-head output matches Reference CPU

- **GIVEN** the same input tensor, `head_count`, rotation `dimension`, base,
  scale, and position offset
- **WHEN** CUDA's multi-head `rope` and Reference CPU's multi-head `rope`
  are each run once
- **THEN** their outputs agree within the declared numeric tolerance

### Requirement: CUDA Provider Baseline

Magnetar SHALL define a CUDA Provider as an optimized, GPU-executing Provider
implementing `magnetar:compute/run`.

#### Scenario: Register CUDA Provider

- **WHEN** Runtime starts with the CUDA Provider enabled
- **THEN** Runtime registers it as a built-in Provider if a compatible CUDA
  driver is discoverable, and registers it with zero Devices otherwise

### Requirement: CUDA Provider Is Not Correctness Baseline

CUDA Provider SHALL preserve Reference CPU's portable Operator semantics and
SHALL NOT be treated as the correctness oracle.

#### Scenario: Divergent output

- **GIVEN** CUDA Provider output differs from Reference CPU beyond declared
  tolerance
- **WHEN** conformance runs
- **THEN** CUDA Provider conformance fails and Reference CPU output remains
  authoritative

### Requirement: CUDA Provider Identity

CUDA Provider SHALL expose stable Provider identity through the same
`ProviderMetadata` contract other Providers use.

#### Scenario: Inspect provider

- **WHEN** Runtime lists Providers with CUDA Provider registered
- **THEN** stable redacted provider metadata is returned, with no native CUDA
  context, driver, or handle value included

### Requirement: Graceful Unavailability Without Compatible Hardware

CUDA Provider SHALL construct successfully and report Provider Health as
`unavailable` (not `failed`) when no compatible CUDA driver or Device is
discoverable, rather than failing Runtime initialization or the build.

#### Scenario: No driver present

- **GIVEN** the host has no CUDA driver installed
- **WHEN** CUDA Provider initializes
- **THEN** it registers with zero Devices and health `unavailable`
- **AND** Runtime initialization continues for other Providers

#### Scenario: CI runner without a GPU

- **GIVEN** `providers/cuda` is built and tested on a runner with no GPU and
  no CUDA Toolkit
- **WHEN** its test suite runs
- **THEN** the crate builds and its unit tests pass by asserting the
  unavailable-health path, without requiring real hardware

### Requirement: CUDA Device Discovery

CUDA Provider SHALL expose its primary compatible CUDA Device (device
ordinal 0) through Runtime-owned Device metadata, including name, compute
capability, and a memory pressure estimate. This baseline targets exactly
one `ModelInstance`/execution bound to exactly one selected GPU Device;
discovering or exposing more than one Device on a multi-GPU host is
explicitly out of scope until a real multi-GPU deployment need drives a
dedicated change (product decision, 2026-09-06). The already-archived
`multi-device-placement` capability (Runtime-owned placement, explicit
`MultiDevicePlacementPlan`) is the target contract for that future work --
this baseline does not implement it, and the two-phase rollout the product
decision named (independent Devices with per-request Runtime placement
first; model/tensor sharding and replication across Devices only if a real
need later emerges) should build on that existing contract rather than
inventing a new one.

#### Scenario: Single GPU available

- **GIVEN** one compatible CUDA-capable GPU is present
- **WHEN** Runtime lists Devices
- **THEN** the primary CUDA Device is visible through Runtime-owned metadata
- **AND** no raw CUDA context, stream, or device pointer is exposed

#### Scenario: Multiple GPUs available

- **GIVEN** more than one compatible CUDA-capable GPU is present on the host
- **WHEN** Runtime lists Devices
- **THEN** only the primary Device (ordinal 0) is visible through
  Runtime-owned metadata
- **AND** no requirement in this baseline depends on any other Device being
  discovered or selectable

### Requirement: CUDA Provider Layout and DType Support

CUDA Provider SHALL declare supported layouts and dtypes explicitly for this
baseline: contiguous layout and f32 only.

No silent dtype or layout conversion SHALL occur.

#### Scenario: Non-contiguous layout requested

- **GIVEN** an invocation requires a non-contiguous layout
- **WHEN** Kernel selection runs
- **THEN** the CUDA Kernel is not selected and Runtime reports a structured
  unsupported-layout error unless explicit conversion is planned

#### Scenario: Non-f32 dtype requested

- **GIVEN** a compute request uses a dtype other than f32
- **WHEN** dispatch is planned
- **THEN** Runtime rejects the CUDA candidate or inserts explicit conversion
  according to policy, without silent coercion

### Requirement: CUDA Provider Kernel Advertisements

CUDA Provider SHALL advertise only implemented Kernels, limited for this
baseline to the `operator-scope` required-now tier needed for the first
decoder path: embedding lookup, RMSNorm, matmul, RoPE (baseline mode only),
causal attention, softmax, SiLU, add, mul, and residual-add.

#### Scenario: Unimplemented kernel

- **GIVEN** flash attention is not implemented by CUDA Provider
- **WHEN** Runtime queries the Kernel Registry
- **THEN** no flash-attention CUDA Kernel is assumed unless explicitly
  advertised

### Requirement: CUDA Kernels Match Reference CPU Semantics

Each advertised CUDA kernel SHALL produce output matching Reference CPU's
output for the same portable Operator within declared numerical tolerance.

#### Scenario: Matmul fixture

- **GIVEN** a small matmul fixture used by Reference CPU conformance
- **WHEN** the same fixture runs through CUDA matmul
- **THEN** output matches Reference CPU's output within tolerance

### Requirement: CUDA Provider Explicit Data Movement

CUDA Provider SHALL require explicit host-to-device and device-to-host data
movement. Host memory inputs SHALL NOT be silently uploaded, and Device-
resident outputs SHALL NOT be silently downloaded. A Kernel invocation whose
input is already device-resident under the same `TensorResourceId` SHALL
NOT trigger a redundant device-to-host-then-host-to-device round trip; the
Provider SHALL reuse the existing device allocation directly.

#### Scenario: Host tensor passed to CUDA kernel

- **GIVEN** an input tensor resides in host memory
- **WHEN** a CUDA Kernel is considered for dispatch
- **THEN** Runtime requires an explicit upload step or rejects dispatch
  according to policy

#### Scenario: Device-resident tensor consumed by a second CUDA kernel

- **GIVEN** a CUDA Kernel's output tensor remains in this Provider's device
  allocation table
- **WHEN** a second CUDA Kernel is dispatched with that same
  `TensorResourceId` as an input
- **THEN** the Provider computes directly against the existing device
  allocation without downloading to host or re-uploading

### Requirement: CUDA Provider Memory Manager Integration

CUDA Provider SHALL allocate real device memory per Tensor Resource directly
(without implementing the Device Memory Pool's soft/hard reservation
contract in this baseline). Admission (byte size, placement, and
`MemoryAllocationOwner`) for an output resource SHALL be performed by
Runtime/caller before submission whenever the caller has already
pre-admitted that resource's final identity; the Provider SHALL then write
its output directly into that identity rather than admitting a second,
Provider-owned allocation for the same resource. When no pre-admission is
found for an output's resource id (e.g. a caller with no independent
resource-identity concept of its own, such as a conformance harness), the
Provider SHALL fall back to admitting and reporting the residency itself,
exactly as before. A Tensor Resource's device allocation SHALL persist
across separate Kernel invocations until this Provider's
`release`/`release_tensor` (or equivalent lifecycle) is called for it; it
SHALL NOT be represented only as host-resident bytes while the Memory
Manager records Device residency for it.

#### Scenario: Caller pre-admits the output resource

- **GIVEN** Runtime has already admitted a `MemoryAllocationRequest` for a
  Kernel's output resource id, under a caller-chosen `MemoryAllocationOwner`
- **WHEN** the CUDA Kernel invocation dispatches
- **THEN** the Provider writes its output directly under that same resource
  id
- **AND** the Provider does not admit a second, Provider-owned allocation
  for it

#### Scenario: No pre-admission found (fallback)

- **GIVEN** a Kernel invocation's output resource id has no existing
  `TensorResidency` record when dispatch begins
- **WHEN** the CUDA Kernel writes that output
- **THEN** the Provider admits it itself, Provider-owned, exactly as this
  baseline's original behavior

#### Scenario: Output consumed by a later invocation

- **GIVEN** a Kernel output tensor was written in a prior invocation
- **WHEN** a later Kernel invocation reads that same `TensorResourceId`
- **THEN** the Provider's device allocation for it is still present and
  valid, without having been silently dropped between calls

### Requirement: CUDA Provider Synchronous Execution

CUDA Provider SHALL execute each submitted operation synchronously to
completion for this baseline, without implementing the Execution Stream
asynchronous completion-token contract.

#### Scenario: Submit compute operation

- **GIVEN** a validated Compute Execution Plan is submitted to CUDA Provider
- **WHEN** the Provider executes it
- **THEN** the call returns only after the kernel has completed or failed,
  with no pending asynchronous completion token

### Requirement: CUDA Provider Error Categories

CUDA Provider failures SHALL use structured error categories mapped from
native CUDA driver/NVRTC errors, and SHALL NOT expose native error codes as
the stable contract.

#### Scenario: Out of device memory

- **GIVEN** a CUDA allocation fails due to insufficient device memory
- **WHEN** Runtime reports the failure
- **THEN** Runtime returns a stable out-of-memory or allocation-failure
  category with the native CUDA error attached only as diagnostics

### Requirement: CUDA Provider Does Not Expose Native Handles

CUDA Provider SHALL NOT expose CUDA contexts, streams, modules, device
pointers, or driver/NVRTC handles through any public Runtime API or
diagnostic surface.

#### Scenario: Diagnostic request

- **GIVEN** CUDA Kernel dispatch fails
- **WHEN** observability records it
- **THEN** Runtime emits redacted structured error metadata with no native
  CUDA pointer or handle value

### Requirement: CUDA Provider Conformance Scope

CUDA Provider SHALL pass the `provider-core`, `provider-compute`, and
`provider-data-movement` conformance profiles for this baseline.
`provider-dynamic-abi` SHALL NOT apply while CUDA Provider remains built-in.
`provider-cancellation` remains deferred to a future change introducing
genuinely asynchronous execution.

#### Scenario: Conformance run

- **GIVEN** CUDA Provider is registered with at least one Device
- **WHEN** the Provider Conformance Suite runs the `provider-core`,
  `provider-compute`, and `provider-data-movement` profiles
- **THEN** the suite reports pass or fail for those profiles specifically,
  without asserting `provider-dynamic-abi` applicability

### Requirement: CUDA Provider Health Reflects Kernel Compilation Failure

CUDA Provider SHALL report Provider Health as `degraded`, not `available`,
when a compatible Device was discovered but this Provider's own Kernels
failed to compile or load.

#### Scenario: Device found, kernel compilation fails

- **GIVEN** a compatible CUDA Device is discovered
- **AND** this Provider's NVRTC kernel compilation fails
- **WHEN** Runtime queries Provider Health
- **THEN** health is reported as `degraded`
- **AND** `execution_api()` reports unavailable, consistent with `degraded`
  rather than contradicting an `available` health report

### Requirement: CUDA Provider Out-Of-Device-Memory Kernel Errors

A CUDA Kernel failure caused by insufficient device memory SHALL be reported
through the Kernel-level out-of-device-memory error category, not a generic
Kernel execution failure.

#### Scenario: Allocation exceeds available device memory

- **GIVEN** a CUDA allocation request exceeds available device memory
- **WHEN** the failure is mapped to a Kernel error
- **THEN** Runtime reports the out-of-device-memory Kernel error category
  with the native CUDA error retained only as diagnostics

### Requirement: CUDA Provider Supports Multi-Step Decode

CUDA Provider SHALL support a generation request requiring more than one decode step. Historical KV concatenation across decode steps SHALL complete without downloading the historical KV tensor to host memory, and the resulting concatenated KV state SHALL remain Device-resident. Device memory held for a decode session's KV state SHALL NOT grow without bound across steps: at most one live allocation per layer, per K/V role, per KV identity (graph edge, pending, committed) SHALL exist at a time, with the previous step's allocation for that same identity released when superseded.

#### Scenario: Multi-token decode succeeds on real hardware

- **GIVEN** a generation request whose `max_tokens` requires more than one decode step, bound to CUDA Provider
- **WHEN** generation runs
- **THEN** it completes successfully and produces the requested number of real, decoded tokens, not an `Unsupported` rejection or an internal residency error

#### Scenario: Historical KV concatenation stays Device-resident

- **GIVEN** a decode step needs to concatenate this step's new K/V with the previous step's historical K/V, both held Device-resident by CUDA Provider
- **WHEN** the concatenation executes
- **THEN** it completes without CUDA Provider producing host-visible bytes for the historical KV tensor, and the concatenated result is itself Device-resident

#### Scenario: Device memory for KV state does not grow unboundedly across steps

- **GIVEN** a decode step's KV pending-write or commit targets a stable resource identity that already holds a previous step's allocation
- **WHEN** the new value is admitted under that same identity
- **THEN** the previous allocation is released and replaced, not left active alongside the new one, so a real multi-token decode (verified: 8 tokens on a synthetic bundle, 16 on the real public checkpoint) does not accumulate KV allocations proportional to step count

#### Scenario: Multi-token CUDA output matches Reference CPU

- **GIVEN** the same real ingested model, tokenizer, and prompt, decoded for the same number of tokens under greedy sampling
- **WHEN** compared between Reference CPU and CUDA Provider
- **THEN** the generated token sequence and decoded text are identical

