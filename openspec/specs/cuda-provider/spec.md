# cuda-provider Specification

## Purpose
TBD - created by archiving change make-first-native-cuda-hot-path-device-resident. Update Purpose after archive.
## Requirements
### Requirement: CUDA RoPE Kernel Supports Native Multi-Head Rotation

The CUDA Provider's `rope` Kernel SHALL accept an optional `head_count`
parameter. When `head_count` is present and greater than `1`, the Kernel
SHALL divide each row into `head_count` equal-width blocks of
`head_width = cols / head_count` columns and, within each block
independently, rotate exactly the first `dimension` columns of that block
(`dimension` MAY be less than `head_width`, partial rotation) using the
same per-pair rotation formula already defined for the single-block case;
`position` is per-row, not per-head. Columns outside a rotated range SHALL
be copied through unchanged, never zeroed. `head_count` absent (or `1`)
SHALL reproduce the existing single-block rotation behavior exactly, with
`dimension` unconstrained by `head_count`'s divisibility rule.

Multi-head rotation SHALL execute as a single Kernel invocation regardless
of `head_count`; the Provider SHALL NOT require one invocation per head.

#### Scenario: Single-block call is unaffected

- **GIVEN** a `rope` Kernel invocation with no `head_count` attribute (or
  `head_count = 1`)
- **WHEN** the CUDA Provider dispatches it
- **THEN** the rotated output is bit-for-bit identical to this Provider's
  pre-existing single-block `rope` behavior

#### Scenario: Multi-head call with different Q and K head counts

- **GIVEN** two `rope` Kernel invocations over the same row count, one with
  `head_count` equal to a model's attention head count and one with
  `head_count` equal to a smaller grouped-query key/value head count
- **WHEN** the CUDA Provider dispatches each
- **THEN** each rotates its own row width using its own `head_width = cols
  / head_count`, independently of the other invocation's head count

#### Scenario: Partial RoPE within a head is supported

- **GIVEN** a `rope` Kernel invocation with `head_count > 1` and a rotation
  `dimension` strictly less than `head_width = cols / head_count`
- **WHEN** the CUDA Provider dispatches it
- **THEN** only the first `dimension` columns of each head's block are
  rotated
- **AND** the remaining columns of that block are copied from input to
  output unchanged, not zeroed

#### Scenario: Multi-head output matches Reference CPU

- **GIVEN** the same input tensor, `head_count`, rotation `dimension`, base,
  scale, and position offset
- **WHEN** CUDA's multi-head `rope` and Reference CPU's multi-head `rope`
  are each run once
- **THEN** their outputs agree within the declared numeric tolerance

