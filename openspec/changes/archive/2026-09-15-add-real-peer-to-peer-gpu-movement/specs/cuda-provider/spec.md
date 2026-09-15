## ADDED Requirements

### Requirement: CUDA Provider Exposes a Real, Explicit Peer-Capability Query

The CUDA Provider SHALL expose a real, explicit query for whether one real GPU Device can directly access another real GPU Device's memory. This query SHALL NOT infer or assume the result from Device similarity (shared vendor, architecture, or memory capacity).

#### Scenario: Two real GPUs are queried for peer capability

- **GIVEN** two real, available CUDA Devices
- **WHEN** the peer-capability query is called for that pair
- **THEN** it returns the real, driver-reported capability, not an assumption derived from the two Devices' own metadata

### Requirement: CUDA Provider Supports Enabling Real Peer Access

The CUDA Provider SHALL support enabling one real GPU Device's context to directly access another real GPU Device's memory, and SHALL NOT perform this without a caller having first obtained a positive result from the peer-capability query for that same pair.

#### Scenario: Peer access is enabled after a positive capability query

- **GIVEN** a peer-capability query that returned true for a real Device pair
- **WHEN** peer access is enabled for that pair
- **THEN** the operation succeeds, including when called more than once for the same pair

### Requirement: CUDA Provider Supports a Real Cross-Device Copy That Never Touches Host Memory

The CUDA Provider SHALL support copying a Tensor Resource's current device allocation directly from one real GPU Device's own storage into another's, via a real device-to-device transfer, without host materialization at any point in the call path.

#### Scenario: A tensor is moved between two real GPUs

- **GIVEN** a tensor resource resident on one real GPU Device, and peer access already enabled between it and a second real GPU Device
- **WHEN** the cross-Device copy is invoked with the second Device's own executor as the destination
- **THEN** the tensor's bytes are readable back from the second Device
- **AND** they are bit-identical to the source
