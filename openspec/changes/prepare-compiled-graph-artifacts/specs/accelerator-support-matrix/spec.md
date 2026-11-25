## ADDED Requirements

### Requirement: Evidence-Gated Support Matrix

The project SHALL maintain a support matrix of Provider × artifact kind × replacement granularity, each cell marked contract, fixture or hardware-qualified. A cell SHALL become hardware-qualified only with a recorded real-hardware run (Device, driver, artifact identities, qualification and measurements), and never on contracts, fixtures or device discovery alone.

#### Scenario: Discovery without compute

- **WHEN** a Provider discovers a Device but has no verified compute path
- **THEN** its cells stay at contract or fixture, and no documentation claims support
