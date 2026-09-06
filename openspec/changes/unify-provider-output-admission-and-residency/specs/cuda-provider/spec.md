## MODIFIED Requirements

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
