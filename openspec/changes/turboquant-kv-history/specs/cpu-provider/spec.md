## ADDED Requirements

### Requirement: CPU Providers Hold TurboQuant KV Stores

The Reference CPU Provider and the optimized CPU Provider SHALL create TurboQuant KV stores and SHALL write each `kv-write` row as its TurboQuant dequantization, so that their attention defines the reference result over a quantized history.

#### Scenario: Rounded rows
- **WHEN** a row is written into a `tq4` store
- **THEN** reading the store returns that row's TurboQuant dequantization
