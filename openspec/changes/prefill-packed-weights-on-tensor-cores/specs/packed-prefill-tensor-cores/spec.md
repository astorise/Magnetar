## ADDED Requirements

### Requirement: Packed Prefill On Tensor Cores

The CUDA Provider SHALL run a matmul against a packed weight with more rows than its packed row Kernels take on Tensor Cores where the Device has them, with the dequantized weight and the activations each split into bfloat16 hi and lo terms and every product but the two low terms' accumulated in f32.

#### Scenario: Same values as the f32 matmul
- **WHEN** 200 rows multiply a packed `Q4_K` weight
- **THEN** the output is within 1e-5 of the largest output of the Reference CPU matmul of the dequantized weight

#### Scenario: Unaligned or without Tensor Cores
- **WHEN** the inner or output dimension is not a multiple of 8, or the Device has no Tensor Cores
- **THEN** the matmul runs against the dequantized f32 weight on the CUDA cores, as before
