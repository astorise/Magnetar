## ADDED Requirements

### Requirement: A Real Chained Computation Can Execute Across Two Physically Distinct Real GPUs

Given two available, distinctly-named CUDA Provider instances bound to two different real GPU ordinals registered into one Runtime, a caller SHALL be able to dispatch a multi-stage computation across both real Devices, with each stage's output explicitly, physically movable to the other Device via an explicit host round trip.

#### Scenario: Two-stage computation across two real GPUs

- **GIVEN** two real, physically distinct GPUs, each with its own registered CUDA Provider instance
- **WHEN** stage one executes on the first real GPU, its result is read back to the host, and admitted fresh into the second real GPU's memory domain for stage two
- **THEN** stage two's real, GPU-computed result matches the expected value for the full chained computation

#### Scenario: Homogeneous real Devices are still tracked as distinct

- **GIVEN** two real GPUs of the identical model and identical memory capacity
- **WHEN** a `DeviceSet` is built from both Devices' own real metadata
- **THEN** the two Devices are recognized as distinct members, not deduplicated by shared architecture/vendor/capacity
