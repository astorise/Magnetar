# rocm-provider Specification

## Purpose
TBD - created by archiving change add-rocm-metal-provider-skeletons-and-real-wgpu-provider. Update Purpose after archive.
## Requirements
### Requirement: ROCm Provider Real Device Discovery

The ROCm Provider SHALL construct successfully on every host, regardless of whether a ROCm runtime is present. At construction, it SHALL attempt to dynamically load a real HIP runtime library from a platform-appropriate, ordered list of candidate library names, and, only if loading succeeds, call the real `hipInit` and `hipGetDeviceCount` entry points to discover a device count.

Any failure at any step (library not found, symbol not found, non-success return code) SHALL result in the Provider reporting no discovered devices and `ProviderHealth::Unavailable`, never a construction error and never a panic.

#### Scenario: No ROCm runtime present

- **GIVEN** a host with no HIP-compatible runtime library installed (true of every machine and CI runner this repository currently has)
- **WHEN** the ROCm Provider is constructed
- **THEN** construction succeeds
- **AND** `health()` reports `ProviderHealth::Unavailable`
- **AND** no devices are reported

#### Scenario: Library present but API call fails

- **GIVEN** a host where the HIP library loads but `hipInit` or `hipGetDeviceCount` returns a non-success code
- **WHEN** the ROCm Provider is constructed
- **THEN** it reports the same graceful `Unavailable` outcome as when the library is entirely absent, not a crash or panic

### Requirement: ROCm Provider Has No Compute Kernels

The ROCm Provider SHALL NOT advertise or implement any compute Kernel. It implements only the `Provider` trait's identity, registration, and health-reporting surface.

#### Scenario: No Kernels advertised

- **GIVEN** a constructed ROCm Provider, available or not
- **WHEN** its Kernel advertisements are queried
- **THEN** the list is empty

