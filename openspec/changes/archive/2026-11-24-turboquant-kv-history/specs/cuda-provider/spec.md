## ADDED Requirements

### Requirement: CUDA TurboQuant KV Stores

The CUDA Provider SHALL store a TurboQuant KV store as codes, norms and offsets, SHALL set the offsets and quantize rows in `kv-write` within step capture, and SHALL compute decode attention on the codes with a rotated query and an inverse rotation of the accumulated values, matching the Reference CPU on the same history within the f32 conformance tolerance.

#### Scenario: Decode on codes
- **WHEN** a decode step attends over a `tq4` history
- **THEN** its output matches the Reference CPU's over the dequantized history within tolerance, with no stored row dequantized
