## MODIFIED Requirements

### Requirement: CUDA Provider Offers Native Half-Precision Elementwise Compute

CUDA Provider SHALL offer real, on-device `F16`/`bfloat16` storage and elementwise `add`/`mul` compute, reachable both as a directly-callable primitive and through the Kernel Registry's standard selection/dispatch contract (`KernelSelectionRequest` -> `KernelRegistry::select` -> `KernelDispatchPlan::from_selection` -> `KernelDispatcher::revalidate` -> `ProviderExecutionApi::submit_kernel`/`complete_kernel`), via a distinct advertised Kernel identity per `add`/`mul` (`CUDA Provider Layout and DType Support`/`CUDA Provider Kernel Advertisements`'s "f32 only" declaration remains accurate for every other advertised Kernel). A half-precision buffer SHALL hold real 2-byte bit patterns of the declared format, not `f32` promoted and labeled, and host `f32` values SHALL convert to/from those bytes only at the explicit upload/download boundary, exactly as CUDA Provider's existing `f32` path already requires explicit data movement. A half-precision resource is NOT required to remain device-resident as half-precision bytes between separate Kernel invocations in this baseline -- unlike this Provider's `f32` device allocation table, which SHALL remain untouched by this requirement.

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

#### Scenario: A Float16 resource selects the native half-precision Kernel through the standard dispatch contract

- **GIVEN** a `KernelSelectionRequest` for the `add` (or `mul`) Operator whose input/output resources declare `Float16` (or `BrainFloat16`) as their `ComputeDType`
- **WHEN** the Kernel Registry selects a candidate and the resulting invocation is dispatched through `ProviderExecutionApi::submit_kernel`/`complete_kernel`
- **THEN** the native half-precision Kernel is selected in preference to the `f32`-only one, and the dispatched result matches the same reference conversion model the direct-call scenarios above already establish
