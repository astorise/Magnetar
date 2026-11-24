## ADDED Requirements

### Requirement: Packed Prefill On Tensor Cores

The CUDA Provider SHALL run a matmul against a packed weight with more rows than its packed row Kernels take on Tensor Cores where the Device has them, with the dequantized weight and the activations each split into three bfloat16 terms and every product of terms down to 2^-24 of the whole accumulated in f32, as precise as its f32 Kernels.

#### Scenario: Same values as the f32 matmul
- **WHEN** 200 rows multiply a packed `Q4_K` weight
- **THEN** the output is within 1e-5 of the largest output of the Reference CPU matmul of the dequantized weight

#### Scenario: Products that cancel
- **WHEN** a gated matmul over packed `Q8_0` weights runs 70 rows whose products cancel
- **THEN** it stays within the f32 Kernels' conformance tolerance

#### Scenario: Unaligned or without Tensor Cores
- **WHEN** the inner or output dimension is not a multiple of 8, or the Device has no Tensor Cores
- **THEN** the matmul runs against the dequantized f32 weight on the CUDA cores, as before
