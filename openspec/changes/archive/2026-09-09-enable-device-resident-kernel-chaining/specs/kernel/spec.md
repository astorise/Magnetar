## ADDED Requirements

### Requirement: Kernel Out-Of-Device-Memory Error Category

`KernelErrorCode` SHALL include a distinct out-of-device-memory category,
separate from generic Kernel execution failure, so a Provider whose Kernel
fails due to insufficient Device memory can report it as a stable,
distinguishable category rather than folding it into an opaque execution
failure.

#### Scenario: Device allocation exhausted during Kernel dispatch

- **GIVEN** a Provider's Kernel invocation fails because Device memory is
  exhausted
- **WHEN** the Provider maps its native error to `KernelErrorCode`
- **THEN** the out-of-device-memory category is used, not a generic
  Kernel-execution-failed category
