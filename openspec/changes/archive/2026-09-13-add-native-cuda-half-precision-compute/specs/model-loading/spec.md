## MODIFIED Requirements

### Requirement: DType Handling

Loading SHALL distinguish storage dtype from compute dtype and validate any conversion or workspace requirements. Both directions of the `F16`/`BF16` <-> `F32` conversion SHALL be available as exact, round-to-nearest-even operations: decoding storage bytes to `F32` (already used to materialize `F16`/`BF16` weights) and encoding an `F32` value back to `F16`/`BF16`, the latter existing to support a future native half-precision execution path that needs to convert host `F32` values into on-device half-precision bytes.

#### Scenario: INT8 storage BF16 compute

Given weights are stored as INT8

And BF16 compute is requested

When loading is planned

Then Runtime validates conversion support and workspace availability.

#### Scenario: F32 to F16/BF16 encoding round-trips every exactly representable value

Given an `F32` value produced by decoding some `F16` (respectively `bfloat16`) bit pattern

When that `F32` value is encoded back to `F16` (respectively `bfloat16`)

Then the re-encoded bit pattern matches the original exactly, for every one of the 65,536 possible bit patterns of that format (NaN payloads excepted, where only "still NaN" is required)
