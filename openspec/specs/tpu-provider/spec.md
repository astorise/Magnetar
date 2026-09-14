# tpu-provider Specification

## Purpose
TBD - created by archiving change add-npu-and-tpu-provider-skeletons. Update Purpose after archive.
## Requirements
### Requirement: TPU Provider Real Device Discovery

The TPU Provider SHALL construct successfully on every host, regardless of whether a real, locally-attached Google Coral Edge TPU is present. At construction, it SHALL attempt to dynamically load a real `libedgetpu` runtime library from a platform-appropriate, ordered list of candidate library names, and, only if loading succeeds, call the real `edgetpu_list_devices` entry point to discover an attached-device count, freeing the returned array via the real `edgetpu_free_devices` entry point before returning.

Any failure at any step (library not found, symbol not found, zero devices) SHALL result in the Provider reporting no discovered devices and `ProviderHealth::Unavailable`, never a construction error and never a panic.

This Provider targets the real, purchasable local Coral Edge TPU accelerator specifically. It SHALL NOT attempt to discover or connect to Google Cloud TPU, a cloud service with no local device to discover.

#### Scenario: No Coral Edge TPU attached

- **GIVEN** a host with no `libedgetpu` runtime library installed, or one with no Coral device attached (true of every machine and CI runner this repository currently has)
- **WHEN** the TPU Provider is constructed
- **THEN** construction succeeds
- **AND** `health()` reports `ProviderHealth::Unavailable`
- **AND** no devices are reported

#### Scenario: Runtime present but no device attached

- **GIVEN** a host where the `libedgetpu` runtime library loads but `edgetpu_list_devices` reports zero devices
- **WHEN** the TPU Provider is constructed
- **THEN** it reports the same graceful `Unavailable` outcome as when the library is entirely absent, not a crash or panic

### Requirement: TPU Provider Has No Compute Kernels

The TPU Provider SHALL NOT advertise or implement any compute Kernel. It implements only the `Provider` trait's identity, registration, and health-reporting surface.

#### Scenario: No Kernels advertised

- **GIVEN** a constructed TPU Provider, available or not
- **WHEN** its Kernel advertisements are queried
- **THEN** the list is empty

