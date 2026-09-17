## ADDED Requirements

### Requirement: CUDA Provider Can Bind to a Specific Real Device Ordinal Under a Distinct Name

The CUDA Provider SHALL support construction bound to a caller-specified real device ordinal, registering under a caller-specified Provider name distinct from the default. The default constructor SHALL be behaviorally identical to binding ordinal 0 under the default name.

#### Scenario: Default construction is unchanged

- **GIVEN** the CUDA Provider's default constructor and its ordinal-0/default-name constructor
- **WHEN** both are constructed on the same host
- **THEN** they report identical availability, health, and Device identity

#### Scenario: A second real GPU ordinal is requested under its own name

- **GIVEN** a host with two or more real, compatible CUDA devices
- **WHEN** the CUDA Provider is constructed bound to ordinal 1 under a distinct Provider name
- **THEN** it reports `ProviderHealth::Available`
- **AND** its one reported Device has an identity distinct from ordinal 0's

#### Scenario: An out-of-range ordinal is requested

- **GIVEN** a host with fewer real CUDA devices than the requested ordinal plus one
- **WHEN** the CUDA Provider is constructed bound to that ordinal
- **THEN** construction succeeds
- **AND** it reports `ProviderHealth::Unavailable`, never a construction error or panic

### Requirement: Two Distinctly-Named CUDA Providers Register Into One Runtime Together

Two CUDA Provider instances bound to two different real device ordinals, each registered under its own distinct Provider name, SHALL both register successfully into the same Runtime.

#### Scenario: Two real GPUs registered together

- **GIVEN** two CUDA Provider instances bound to two different real device ordinals under two distinct names
- **WHEN** both are registered into the same Runtime
- **THEN** Runtime construction succeeds
- **AND** both Providers' Devices and Kernels are present in that Runtime
