## ADDED Requirements

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

CUDA Provider SHALL expose each discoverable compatible CUDA Device through
Runtime-owned Device metadata, including name, compute capability, and a
memory pressure estimate.

#### Scenario: Single GPU available

- **GIVEN** one compatible CUDA-capable GPU is present
- **WHEN** Runtime lists Devices
- **THEN** at least one CUDA Device is visible through Runtime-owned metadata
- **AND** no raw CUDA context, stream, or device pointer is exposed

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
resident outputs SHALL NOT be silently downloaded.

#### Scenario: Host tensor passed to CUDA kernel

- **GIVEN** an input tensor resides in host memory
- **WHEN** a CUDA Kernel is considered for dispatch
- **THEN** Runtime requires an explicit upload step or rejects dispatch
  according to policy

### Requirement: CUDA Provider Memory Manager Integration

CUDA Provider SHALL allocate device memory per Tensor Resource directly
(without implementing the Device Memory Pool's soft/hard reservation
contract in this baseline) and SHALL report resulting residency and Resource
Affinity to Runtime Memory Manager.

#### Scenario: Kernel output tensor

- **GIVEN** a CUDA Kernel writes an output tensor
- **WHEN** dispatch completes
- **THEN** Memory Manager records Device residency and Provider-pinned
  affinity for that output

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

CUDA Provider SHALL pass the `provider-core` and `provider-compute`
conformance profiles for this baseline. `provider-dynamic-abi` SHALL NOT
apply while CUDA Provider remains built-in. `provider-cancellation` and full
`provider-data-movement` profiles are deferred to a future change introducing
the Execution Stream extension.

#### Scenario: Conformance run

- **GIVEN** CUDA Provider is registered with at least one Device
- **WHEN** the Provider Conformance Suite runs the `provider-core` and
  `provider-compute` profiles
- **THEN** the suite reports pass or fail for those profiles specifically,
  without asserting `provider-dynamic-abi` applicability
