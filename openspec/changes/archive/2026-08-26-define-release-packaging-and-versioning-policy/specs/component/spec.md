## ADDED Requirements

### Requirement: Component Contract Release Status

Release metadata SHALL declare Component Contract compatibility status.

#### Scenario: Component API status

Given `v0.1` release notes are generated

When Component Contract is listed

Then it is marked baseline, experimental, or deferred as appropriate.

---

### Requirement: Component Engine Feature Status

Release metadata SHALL document Component Engine feature flags and supported
platforms.

#### Scenario: Wasmtime feature

Given Wasmtime Component Engine is feature-gated

When release docs are generated

Then support status and limitations are documented.