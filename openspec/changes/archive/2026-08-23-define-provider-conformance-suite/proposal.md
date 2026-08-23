# Define Provider Conformance Suite

## Why

Magnetar Providers are trusted native execution extensions.

They implement Runtime Capabilities and expose Devices used for local inference
execution.

Previous changes clarified:

- Providers are the native extension mechanism
- Components are portable WASM Components
- Components import Capabilities, not Providers
- Runtime owns Provider and Device resolution
- Resource Affinity is authoritative
- Compute descriptors are portable and do not expose Provider/Device selectors
- Providers report health, readiness, pressure, admission, Device status, and
  Capability status
- dynamic Provider loading requires an explicit ABI policy
- broad filesystem/network/Git/secrets/tool authority is outside Magnetar
- Magnetar is scoped to inference execution

The Runtime now needs a way to determine whether a Provider implementation
actually conforms to Magnetar's Provider contract.

A Provider should not be considered compatible merely because it can be loaded
or registered.

A Provider must demonstrate that it behaves correctly across:

- metadata
- Capability advertisements
- Device metadata
- status reporting
- readiness and pressure
- execution behavior
- error mapping
- cancellation
- Resource Affinity
- data movement
- observability
- lifecycle
- ABI loading where applicable

This change defines a Provider Conformance Suite.

The suite provides reusable tests that Provider implementations can run to prove
compatibility with Magnetar.

## What Changes

This change introduces a formal Provider Conformance Suite.

The suite SHALL validate Provider behavior against Magnetar's Provider
contract.

The suite SHALL be usable for:

- built-in Providers
- test Providers
- development Providers
- dynamic-library Providers
- future CPU Provider
- future CUDA Provider
- future Metal Provider
- future OpenVINO Provider
- future QNN Provider
- temporary Candle Provider

The suite SHALL focus on Magnetar inference-runtime behavior.

It SHALL NOT test client-side filesystem, Git, shell, workspace, secret, or
network tools because those are outside Magnetar Provider scope.

### Conformance Target

A conformance target is a Provider implementation plus the configuration needed
to instantiate it in a test Runtime.

A target MAY be:

```text
built-in Provider
dynamic Provider library
test fixture Provider
development Provider
```

The suite SHALL treat all targets through the same Provider-facing Runtime
contract where possible.

### Required Conformance Areas

A Provider SHALL be tested in at least these areas:

- Provider metadata
- Provider identity
- Capability advertisement
- Device metadata
- Provider lifecycle
- Provider status
- Device status
- Capability status
- execution admission
- Compute execution where supported
- data movement where advertised
- Resource Affinity behavior
- Provider-owned resource handling
- cancellation behavior
- error mapping
- observability integration
- ABI loading behavior for dynamic Providers

### Capability-Specific Conformance

The suite SHALL support Capability-specific test groups.

For a Provider advertising:

```text
magnetar:compute/run
```

the Provider SHALL pass Compute conformance tests.

For future Capabilities such as:

```text
magnetar:generation/...
magnetar:model/...
```

additional conformance groups MAY be added.

A Provider SHALL not advertise a Capability unless it passes the corresponding
conformance requirements for that Capability.

### Metadata Conformance

Provider metadata SHALL be validated.

The suite SHALL verify:

- ProviderId is valid
- ProviderId is stable across runs when configured identically
- Provider name is present
- Provider version is valid
- vendor or implementation metadata is well-formed
- Runtime compatibility is valid
- feature flags are valid
- metadata does not claim unsupported Capabilities
- metadata does not expose native handles

### Capability Advertisement Conformance

Capability advertisements SHALL be tested.

The suite SHALL verify that:

- advertised Capability identifiers are valid
- advertised versions are valid
- unsupported major versions are not claimed
- operation support is consistent
- data movement support is consistent
- advertised Device requirements are valid
- advertised memory requirements are valid
- advertised features match behavior

Advertisement conformance is mandatory because Runtime Resolution depends on
advertisement correctness.

### Device Metadata Conformance

Device metadata SHALL be tested.

The suite SHALL verify:

- DeviceId validity
- Device type validity
- Provider ownership
- memory metadata correctness where provided
- feature metadata correctness where provided
- duplicate Device detection
- absence of raw native handles from public metadata

### Status Conformance

Provider status reporting SHALL be tested against the refined status model.

The suite SHALL verify:

- lifecycle state is valid
- health state is valid
- readiness state is valid
- pressure state is valid
- admission decision is valid
- freshness/TTL is present where required
- stale status behavior is testable
- Device-level status is consistent
- Capability-level status is consistent
- saturated is distinct from failed
- healthy is distinct from ready
- not-ready is distinct from unhealthy

### Execution Conformance

Execution conformance SHALL test Provider behavior for supported operations.

For Compute Providers, the suite SHALL include small deterministic Compute
fixtures.

The suite SHOULD test:

- successful operation execution
- unsupported operation rejection
- invalid input rejection
- invalid dtype rejection
- invalid layout rejection
- invalid shape rejection
- memory planning rejection
- Provider not-ready rejection
- Provider saturated rejection
- Provider draining rejection
- execution failure after admission
- stable output metadata
- stable error mapping

The suite SHALL avoid requiring real GPU hardware in default CI.

Hardware-specific tests MAY be optional or feature-gated.

### Numerical Correctness

Where a Provider performs numerical Compute, the suite SHALL define expected
numeric tolerance.

Tolerance MAY depend on:

- dtype
- operation family
- backend implementation
- quantization mode
- deterministic versus approximate mode
- hardware behavior

A Provider SHALL declare the relevant mode.

The suite SHALL verify that output is within the accepted tolerance for the
declared mode.

### Resource Affinity Conformance

The suite SHALL test Provider resource binding behavior.

It SHALL verify:

- Provider-owned resources remain associated with their Provider
- Device-bound resources remain associated with their Device
- dependent operations preserve binding
- incompatible placement is rejected unless explicit movement is supported
- draining does not silently migrate resources
- Resource Affinity metadata cannot be forged by Component input
- Runtime-owned affinity state remains authoritative

### Data Movement Conformance

For Providers advertising data movement support, the suite SHALL test:

- upload
- download
- copy
- materialize
- transfer
- dtype conversion
- placement conversion
- host staging forbidden
- host staging permitted
- host staging denied by policy
- explicit movement failure
- output descriptor correctness
- affinity after movement

A Provider SHALL not advertise data movement it cannot perform according to
Magnetar semantics.

### Cancellation Conformance

If a Provider advertises cancellation support, the suite SHALL test:

- cancellation before execution
- cancellation during execution
- cancellation after completion
- unsupported cancellation
- cancellation failure
- idempotent cancellation where required
- cleanup after cancellation
- status after cancellation

If a Provider does not support cancellation, it SHALL report that fact
explicitly and map requests to stable errors.

### Error Mapping Conformance

The suite SHALL verify that Provider errors map to stable Magnetar errors.

Error tests SHALL cover:

- Provider not ready
- Provider draining
- Provider saturated
- Device unavailable
- Capability not ready
- invalid request
- unsupported operation
- unsupported dtype
- unsupported layout
- memory allocation failure
- out of memory
- execution failure
- resource invalid
- cancellation unsupported
- cancellation failed
- internal Provider error

Provider-specific diagnostics MAY be included but SHALL be redacted and stable
enough for users.

### Observability Conformance

The suite SHALL verify that Provider operations emit or support expected
Runtime observations without controlling execution semantics.

Tests SHALL ensure:

- execution observations can be emitted
- failure observations can be emitted
- cancellation observations can be emitted
- status observations can be emitted
- observability failure does not alter execution result
- native handles are not exposed in observations

### Lifecycle Conformance

The suite SHALL test Provider lifecycle behavior.

It SHALL cover:

- initialization
- registration
- readiness transition
- drain start
- drain completion
- failure transition
- shutdown
- resource cleanup
- post-shutdown rejection

### ABI Conformance For Dynamic Providers

Dynamic Providers SHALL pass ABI conformance tests.

The suite SHALL validate:

- factory symbol
- ABI version
- descriptor structure
- required function pointers
- metadata function
- Capability advertisement function
- Device metadata function
- status function
- execution function or table
- memory release functions
- destroy function
- no Rust trait-object dynamic ABI
- no unwind across ABI
- error normalization

### Conformance Profiles

The suite SHALL support profiles.

Initial profiles SHOULD include:

```text
provider-core
provider-compute
provider-data-movement
provider-cancellation
provider-observability
provider-dynamic-abi
```

A Provider must pass the profiles corresponding to the features it advertises.

For example:

```text
CPU Provider:
    provider-core
    provider-compute
    provider-data-movement
    provider-observability

CUDA Provider:
    provider-core
    provider-compute
    provider-data-movement
    provider-cancellation
    provider-observability
    provider-dynamic-abi if loaded dynamically
```

### Conformance Report

The suite SHALL produce a structured conformance report.

The report SHOULD include:

- Provider identity
- Provider version
- Runtime version
- conformance suite version
- selected profiles
- passed tests
- failed tests
- skipped tests
- unsupported optional features
- diagnostics
- timestamp

The report SHOULD be machine-readable.

A JSON report format MAY be used.

### CI Integration

Built-in and mock Providers SHALL run conformance profiles in CI.

Hardware-dependent Providers MAY run conformance in optional jobs.

Dynamic ABI fixtures SHALL run in CI where supported by the platform.

### Non-Conformant Providers

A Provider that fails required conformance tests SHALL not be marked production
compatible.

Development mode MAY allow loading non-conformant Providers for local testing,
but the Runtime SHALL expose non-conformance clearly.

### Versioning

The conformance suite SHALL have its own version.

A Provider passing one conformance suite version SHALL not automatically be
assumed to pass future versions.

Provider compatibility documentation SHALL identify which conformance suite
version was passed.

## Non-Goals

This change does not:

- implement a real CUDA Provider
- implement a real Metal Provider
- implement a real OpenVINO Provider
- implement a real QNN Provider
- implement model inference
- define model artifact conformance
- define Component conformance suite
- define Tachyon conformance
- require real GPU hardware in default CI
- define external certification authority
- replace normal unit and integration tests
- allow Providers to bypass Runtime policy
- make Providers sandboxed
- make Provider ABI portable across all languages automatically

## Impact

Magnetar gains a clear compatibility bar for Providers.

A Provider becomes compatible because it passes a defined suite, not because it
happens to compile or register.

This prepares the project for real Provider implementations such as:

- CPU
- CUDA
- Metal
- OpenVINO
- QNN
- Candle temporary Provider

and reduces the risk of Provider-specific behavior silently violating Runtime
contracts.