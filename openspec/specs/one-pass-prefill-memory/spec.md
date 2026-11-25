# one-pass-prefill-memory Specification

## Purpose
This specification defines how a one-pass prefill bounds its working memory: the logits it materializes, the operands of its split matmuls, its grouped expert products and its tiled attention, so that a long prompt prefills on the Devices the model already fits.
## Requirements
### Requirement: One Logits Row Per Prefill

The Runtime SHALL insert the logits row gather in every prefill graph it prepares outside tensor parallelism, so that the LM head of a prompt prefilled in one pass computes the sampled row's logits only.

#### Scenario: A long prompt in one pass
- **WHEN** a prompt that cannot be chunked is prefilled in one pass
- **THEN** the prefill's logits hold one row, the prompt's last, and the generation is the one every row's logits gave

#### Scenario: Buckets keep their padding rule
- **WHEN** a prefill graph at the prompt's own length is neither a bucket's nor the chunk's
- **THEN** its KV edges may have readers other than attention, as before

### Requirement: Bounded Split Matmul Operands

The CUDA Provider SHALL split a long prefill matmul's A into bfloat16 terms at most 2,048 rows at a time, and SHALL hold a gated matmul's gate and up products over packed weights for at most 2,048 rows at a time, every output value unchanged.

#### Scenario: Rows across blocks
- **WHEN** a 4,500-row matmul, matmul-residual or gated-matmul runs against packed weights
- **THEN** its rows in the first block, across a block boundary and in the last block equal, bit for bit, those of the same product over those rows alone

### Requirement: Grouped Expert Prefill

The CUDA Provider SHALL serve a `moe-experts` prefill over packed `Q8_0` experts with its (token, expert) pairs grouped by expert, reading each expert's weights once per up to eight of its pairs, at most 1,024 tokens at a time, with the values of the per-pair Kernels.

#### Scenario: Same values
- **WHEN** a prefill of 65, 300 or 2,500 tokens runs over every expert or a share of them, with SiLU or GELU gates and a per-expert scale
- **THEN** its output equals the per-pair Kernels' output bit for bit

### Requirement: Tiled Attention With Narrower Values

The CUDA Provider SHALL run a prefill attention with 192-dimension keys and 128-dimension values on its tiled prefill Kernel.

#### Scenario: Expanded latent attention
- **WHEN** a prefill attention over 192-dimension keys and 128-dimension values runs with several query tiles, a window, grouped heads or queries after a longer history
- **THEN** it matches the Reference CPU within the attention tolerance
