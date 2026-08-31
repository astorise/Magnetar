## ADDED Requirements

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
