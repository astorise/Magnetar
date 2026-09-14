## ADDED Requirements

### Requirement: Metal Provider Is Unconditionally Unavailable

The Metal Provider SHALL construct successfully on every host and SHALL report `ProviderHealth::Unavailable` unconditionally, on every platform, with no platform-conditional code path claiming otherwise. This reflects that no macOS development or CI environment exists in this repository's tooling to implement or verify real Metal FFI against; an unconditional, honestly-labeled placeholder is required rather than an unverified `#[cfg(target_os = "macos")]` implementation.

#### Scenario: Unavailable regardless of host platform

- **GIVEN** a constructed Metal Provider, on any operating system
- **WHEN** its health is queried
- **THEN** it reports `ProviderHealth::Unavailable`
- **AND** no devices are reported

### Requirement: Metal Provider Documents Its Real Future Scope

The Metal Provider's documentation SHALL state why it exists despite `providers/wgpu` covering Apple GPU access today: `wgpu`/WGSL cannot reach Apple Silicon's `simdgroup_matrix` matrix-multiply-accumulate instructions or route through Metal Performance Shaders to the AMX coprocessor, which matters for compute-bound workloads (prefill) though not for memory-bound ones (decode). It SHALL invite contributors with real Apple Silicon hardware to implement and verify it.

#### Scenario: Documentation states the relationship to providers/wgpu

- **GIVEN** this crate's README
- **WHEN** read by a future contributor
- **THEN** it explains that `providers/wgpu` is the current real path to Apple GPUs
- **AND** it explains the specific compute-bound gap (`simdgroup_matrix`/AMX/MPS) this crate remains real future work for
- **AND** it explicitly invites contribution from someone with real Apple Silicon hardware
