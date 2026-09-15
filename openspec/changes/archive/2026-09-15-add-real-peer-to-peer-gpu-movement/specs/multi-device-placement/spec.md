## ADDED Requirements

### Requirement: Real Peer Access, Once Confirmed Available, Can Genuinely Move a Resource Without Host Staging

When two real Devices report genuine peer-access capability, a caller SHALL be able to move a Tensor Resource between them without host staging, and this movement SHALL be representable with `HostStagingPolicy::Forbid` truthfully -- distinct from a host-staged crossing, which SHALL be represented with `HostStagingPolicy::Permit`.

#### Scenario: A real peer movement is represented as Forbid

- **GIVEN** a real cross-Device movement that used direct peer-to-peer device memory access, never touching host memory
- **WHEN** a `StageMovementEdge` is built to describe it
- **THEN** its `host_staging_policy` is `Forbid`, honestly reflecting that no host staging occurred

#### Scenario: Peer capability absent falls back to explicit host staging

- **GIVEN** two real Devices whose peer-capability query returns false
- **WHEN** a caller needs to move a resource between them
- **THEN** the caller uses an explicit host-staged crossing instead, represented with `HostStagingPolicy::Permit`, never a silent assumption of peer access
