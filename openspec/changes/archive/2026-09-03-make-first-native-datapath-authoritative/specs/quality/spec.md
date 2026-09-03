## ADDED Requirements

### Requirement: Coverage Measures Production Source
The documented production coverage baseline SHALL exclude test-only Rust code from the measured production source scope.

#### Scenario: Test-only code moves or grows
- **WHEN** only test-only code changes
- **THEN** the production coverage baseline is not artificially improved.

#### Scenario: Coverage baseline is updated
- **WHEN** the coverage baseline is regenerated
- **THEN** its documented exclusions match the coverage tool output.
