## RENAMED Requirements

- FROM: `### Requirement: ROCm Provider Has No Compute Kernels`
- TO: `### Requirement: ROCm Provider Advertises Only Qualified Compute Profiles`

## MODIFIED Requirements

### Requirement: ROCm Provider Advertises Only Qualified Compute Profiles

The ROCm Provider SHALL advertise no compute Kernels in its discovery-only profile. A separately enabled compute profile MAY advertise only operators, dtypes, layouts and Device versions proven through real target hardware conformance and real-checkpoint validation. Missing drivers, unsupported operations and unqualified profiles SHALL remain explicitly unavailable or Unsupported. Native library loading and delegated execution SHALL remain inside the Provider; qualification of one vendor/profile SHALL NOT imply another profile is supported.

#### Scenario: Discovery without qualified compute

- **WHEN** a real Device is enumerated but no compute profile has passed target qualification
- **THEN** its compute advertisement is empty and no LLM compute readiness is inferred

#### Scenario: Qualified compute is enabled

- **GIVEN** real target evidence for a specific operator/dtype/shape profile
- **WHEN** that compatible profile is enabled
- **THEN** only its qualified Kernels are advertised through generic dispatch

#### Scenario: Driver or delegated operator is unavailable

- **WHEN** the required runtime or operator support is missing
- **THEN** construction remains safe and compute reports unavailable or Unsupported without a false success
