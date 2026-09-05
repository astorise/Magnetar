## ADDED Requirements

### Requirement: Documentation Quality Gate Covers Runtime Features
Quality checks SHALL keep documentation builds green for the workspace default feature set and the Wasmtime Component Engine feature set.

#### Scenario: Default docs
- **WHEN** CI runs documentation quality checks
- **THEN** `cargo doc --workspace --no-deps` succeeds without global warning suppression added solely for CI.

#### Scenario: Wasmtime docs
- **WHEN** CI validates Component Runtime documentation
- **THEN** `cargo doc -p magnetar-runtime --no-deps --features wasmtime-component-engine` or an equivalent job succeeds.
