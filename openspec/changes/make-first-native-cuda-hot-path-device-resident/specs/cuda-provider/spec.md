## ADDED Requirements

### Requirement: CUDA RoPE Kernel Supports Native Multi-Head Rotation

The CUDA Provider's `rope` Kernel SHALL accept a `head_count` parameter and
rotate each of `head_count` contiguous, equal-width column blocks per row
independently, using the same per-block rotation formula already defined
for a single block. `head_count` not provided (or `1`) SHALL reproduce the
existing single-block rotation behavior exactly.

Multi-head rotation SHALL execute as a single Kernel invocation regardless
of `head_count`; the Provider SHALL NOT require one invocation per head.

#### Scenario: Single-block call is unaffected

- **GIVEN** a `rope` Kernel invocation with no `head_count` attribute (or
  `head_count = 1`)
- **WHEN** the CUDA Provider dispatches it
- **THEN** the rotated output is bit-for-bit identical to this Provider's
  pre-existing single-block `rope` behavior

#### Scenario: Multi-head call rotates each head independently

- **GIVEN** a `rope` Kernel invocation with `head_count > 1` and a row width
  equal to `head_count` times the per-head rotation width
- **WHEN** the CUDA Provider dispatches it
- **THEN** each head's column block is rotated using that head's own local
  column offset within the block, with position and frequency computed the
  same way for every head in a row
- **AND** exactly one Kernel invocation performs the rotation for all heads

#### Scenario: Multi-head output matches Reference CPU

- **GIVEN** the same input tensor, `head_count`, rotation width, base, scale,
  and position offset
- **WHEN** CUDA's multi-head `rope` and Reference CPU's multi-head `rope` are
  each run once
- **THEN** their outputs agree within the declared numeric tolerance
