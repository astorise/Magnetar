## RENAMED Requirements

- FROM: `### Requirement: Metal Provider Is Unconditionally Unavailable`
- TO: `### Requirement: Metal Provider Availability Requires Hardware Qualification`
- FROM: `### Requirement: Metal Provider Documents Its Real Future Scope`
- TO: `### Requirement: Metal Provider Documents Its Measured Scope`

## MODIFIED Requirements

### Requirement: Metal Provider Availability Requires Hardware Qualification

The native Metal Provider SHALL construct safely on every host. It SHALL report Unavailable and expose no Devices on non-macOS systems, when required libraries or Devices are absent, or when no real-Apple-hardware-qualified profile is enabled. On a supported macOS host it MAY report a narrowly qualified experimental or production profile after real hardware conformance and comparison with WGPU. Platform-conditional compilation alone SHALL NOT constitute proof of support.

#### Scenario: Non-macOS or unqualified environment

- **WHEN** the Metal Provider is constructed without an enabled qualified Apple profile
- **THEN** it reports Unavailable with no Devices and does not panic

#### Scenario: Qualified Apple profile

- **GIVEN** real Apple Silicon conformance evidence and an explicitly enabled compatible profile
- **WHEN** required Device and library initialization succeeds
- **THEN** only that profile is advertised with its actual maturity and supported operations

#### Scenario: Initialization failure on supported macOS

- **WHEN** native Device or library initialization fails
- **THEN** the Provider remains unavailable without claiming compute capability

### Requirement: Metal Provider Documents Its Measured Scope

Metal documentation SHALL distinguish WGPU's existing Apple GPU access from native specialized prefill experiments, explain any actually verified simdgroup_matrix or MPS use without assuming AMX access, and publish measured support limits and the real Apple hardware used. Missing hardware SHALL be documented as an unvalidated proposal, not native compute support.

#### Scenario: POC has no target hardware evidence

- **WHEN** documentation describes a native prefill idea without a real Apple run
- **THEN** it labels it proposed/experimental and retains WGPU as the established path within WGPU's own qualified scope

#### Scenario: Specialized path is measured

- **WHEN** a native profile passes comparison with WGPU on actual Apple hardware
- **THEN** documentation records its numerical, memory and latency evidence and does not generalize to untested Devices
