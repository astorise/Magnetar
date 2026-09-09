## MODIFIED Requirements

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
contract in this baseline) and SHALL report resulting residency and Resource
Affinity to Runtime Memory Manager. A Tensor Resource's device allocation
SHALL persist across separate Kernel invocations until this Provider's
`release`/`release_tensor` (or equivalent lifecycle) is called for it; it
SHALL NOT be represented only as host-resident bytes while the Memory
Manager records Device residency for it.

#### Scenario: Kernel output tensor

- **GIVEN** a CUDA Kernel writes an output tensor
- **WHEN** dispatch completes
- **THEN** Memory Manager records Device residency and Provider-pinned
  affinity for that output
- **AND** the Provider's own storage for that output is a real device
  allocation, not host-resident bytes

#### Scenario: Output consumed by a later invocation

- **GIVEN** a Kernel output tensor was written in a prior invocation
- **WHEN** a later Kernel invocation reads that same `TensorResourceId`
- **THEN** the Provider's device allocation for it is still present and
  valid, without having been silently dropped between calls

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

## ADDED Requirements

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
