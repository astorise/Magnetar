## ADDED Requirements

### Requirement: Kernel Source WIT Package

The Runtime SHALL define `magnetar:kernel-source@0.1.0` as an additive WIT package with an interface `source` (`list-bundles`, `read-chunk`) and a world `kernel-package` that exports it and imports nothing. No existing WIT package or signature SHALL change.

#### Scenario: Existing Components still link

- **WHEN** the package is added
- **THEN** every existing model, encoder and observability Component links and runs unchanged

#### Scenario: The package carries no native types

- **WHEN** the WIT is validated
- **THEN** it contains no pointer, handle, Provider, Device or path type, only digests, sizes, labels and bytes
