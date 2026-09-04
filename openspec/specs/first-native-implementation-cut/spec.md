# first-native-implementation-cut Specification

## Purpose
TBD - created by archiving change define-first-native-model-implementation-and-conformance-cut. Update Purpose after archive.
## Requirements
### Requirement: Implementation Shall Follow Dependency Gates

The first native model implementation SHALL satisfy lower-layer gates before
claiming higher-layer integration completion.

#### Scenario: Qwen output exists but Registry bypassed

Given a Qwen-specific implementation can produce correct output

When Kernel Registry/Provider path is not traversed

Then first-native implementation cut is not complete.

### Requirement: Architecture Freeze #1

Acceptance of this change SHALL freeze mandatory first-profile architecture
until a blocker class requiring architectural correction is discovered.

#### Scenario: New optimization idea appears

Given implementation is underway

When a new autotuning capability is proposed

Then it SHALL NOT block the baseline unless required for correctness/security/
implementability.

### Requirement: First Baseline Prioritizes Correctness

Performance optimization SHALL not be required for the first implementation
baseline.

#### Scenario: Scalar CPU MatMul

Given scalar implementation is correct and conformant

When baseline is evaluated

Then lack of SIMD does not fail the cut.

### Requirement: Lower Layer Unit Success Is Not E2E Success

A successful Kernel or Provider test SHALL not satisfy Runtime/CLI E2E
conformance by itself.

#### Scenario: Attention Kernel passes

Given Attention unit tests are green

But Runtime model execution is missing

When milestone status is evaluated

Then full vertical slice remains incomplete.

### Requirement: Temporary Bypasses Are Not Final Conformance

Migration shortcuts SHALL not participate in final first-native E2E.

#### Scenario: Temporary fake logits helper exists

Given helper is still useful in focused tests

When native E2E runs

Then helper is not used.

### Requirement: Done Requires Structural Evidence

Final output alone SHALL not be sufficient proof of architectural conformance.

#### Scenario: Expected token produced

Given output matches golden

But Component/Registry evidence is absent

When cut is validated

Then conformance remains incomplete.

### Requirement: Final Cut Has No P0 Bypass
Architecture Freeze #1 SHALL NOT be declared complete while any F01-F06 bypass remains in the normal first-native path.

#### Scenario: Bypass inventory blocks final cut
- **WHEN** `RuntimeGenerationExecutor`, production logits injection, CLI E2E harness dependency, direct Reference CPU model execution, Rust fixture graph fallback, or full-history decode remains active
- **THEN** the bypass inventory marks the item removal-required and final conformance is blocked.

### Requirement: Bypass Markers Reflect Code Reality
Bypass markers SHALL only be cleared when the corresponding bypass is absent from the normal first-native path.

#### Scenario: Marker cannot hide implementation gap
- **WHEN** code still uses a migration shortcut
- **THEN** the marker remains present with a removal-required disposition.

