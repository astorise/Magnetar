# npu-provider Specification

## Purpose
TBD - created by archiving change add-npu-and-tpu-provider-skeletons. Update Purpose after archive.
## Requirements
### Requirement: NPU Provider Real Device Discovery

The NPU Provider SHALL construct successfully on every host, regardless of whether a Level Zero NPU/VPU-class driver is present. At construction, it SHALL attempt to dynamically load a real oneAPI Level Zero loader library from a platform-appropriate, ordered list of candidate library names, and, only if loading succeeds, call the real `zeInit` entry point with the `ZE_INIT_FLAG_VPU_ONLY` flag (scoping discovery to NPU/VPU-class devices, not GPUs) followed by the real `zeDriverGet` entry point to discover a driver count.

Any failure at any step (library not found, symbol not found, non-success return code, zero drivers) SHALL result in the Provider reporting no discovered devices and `ProviderHealth::Unavailable`, never a construction error and never a panic.

#### Scenario: No Level Zero NPU driver present

- **GIVEN** a host with no Level Zero loader library installed, or one with no NPU/VPU-class driver registered (true of every machine and CI runner this repository currently has)
- **WHEN** the NPU Provider is constructed
- **THEN** construction succeeds
- **AND** `health()` reports `ProviderHealth::Unavailable`
- **AND** no devices are reported

#### Scenario: Loader present but initialization fails

- **GIVEN** a host where the Level Zero loader library loads but `zeInit(ZE_INIT_FLAG_VPU_ONLY)` or `zeDriverGet` returns a non-success result
- **WHEN** the NPU Provider is constructed
- **THEN** it reports the same graceful `Unavailable` outcome as when the library is entirely absent, not a crash or panic

### Requirement: NPU Provider Has No Compute Kernels

The NPU Provider SHALL NOT advertise or implement any compute Kernel. It implements only the `Provider` trait's identity, registration, and health-reporting surface.

#### Scenario: No Kernels advertised

- **GIVEN** a constructed NPU Provider, available or not
- **WHEN** its Kernel advertisements are queried
- **THEN** the list is empty

