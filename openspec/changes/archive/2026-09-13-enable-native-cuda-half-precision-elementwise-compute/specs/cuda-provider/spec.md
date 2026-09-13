## ADDED Requirements

### Requirement: CUDA Provider Offers Native Half-Precision Elementwise Compute

CUDA Provider SHALL offer real, on-device `F16`/`bfloat16` storage and elementwise `add`/`mul` compute as a directly-callable primitive, distinct from -- and not yet participating in -- the advertised Kernel Registry surface `CUDA Provider Layout and DType Support`/`CUDA Provider Kernel Advertisements` govern (those requirements' "f32 only" declaration remains accurate for the advertised, planner-selectable surface; wiring compute-dtype selection into the Runtime's planner is separate, future work). A half-precision buffer SHALL hold real 2-byte bit patterns of the declared format, not `f32` promoted and labeled, and host `f32` values SHALL convert to/from those bytes only at the explicit upload/download boundary, exactly as CUDA Provider's existing `f32` path already requires explicit data movement.

#### Scenario: F16 elementwise add matches the reference conversion model

- **GIVEN** two `f32` host tensors of equal shape
- **WHEN** both are uploaded as `F16`, added via the native half-precision `add` kernel, and downloaded
- **THEN** the result matches, element-wise, decoding-then-re-encoding each already-half-precision-rounded input through the same round-to-nearest-even algorithm and re-rounding the sum

#### Scenario: BF16 elementwise mul matches the reference conversion model

- **GIVEN** two `f32` host tensors of equal shape
- **WHEN** both are uploaded as `bfloat16`, multiplied via the native half-precision `mul` kernel, and downloaded
- **THEN** the result matches, element-wise, the same reference conversion model as the F16 scenario, using `bfloat16`'s own round-to-nearest-even algorithm

#### Scenario: Half-precision buffers of different dtypes cannot be combined

- **GIVEN** one `F16` half-precision buffer and one `bfloat16` half-precision buffer
- **WHEN** the native half-precision `add` or `mul` kernel is invoked with both
- **THEN** the call is rejected before any kernel launch, without silent coercion
