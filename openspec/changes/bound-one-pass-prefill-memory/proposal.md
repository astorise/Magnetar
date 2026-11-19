## Why

A model whose prefill cannot run in chunks (`chunk-long-prefills`: an
expanded latent attention reads K and V computed from its own rows while its
history holds the latent rows) prefills a long prompt in one pass. Every
activation then holds every prompt row. DeepSeek-V2-Lite-Chat Q8_0 on two
RTX 3060 (12 GB) shows where that breaks
(`cache-latent-attention-history`, task 6.7):

- with the first GPU filled with weights (85% of its memory), a
  4,096-token prefill ran out of its memory (run 44358602056);
- with the layers balanced across both GPUs (3,890 and 2,740 MiB free after
  the load), 4,096 tokens run, and 16,384 tokens run out of the first GPU's
  memory in layer 0's gated MLP (run 44374637064).

Four costs of a one-pass prefill grow with the prompt beyond what it needs:
the LM head computes every row's logits though only the last row is sampled
(16,384 rows of a 102,400-token vocabulary are 6.7 GB); a gated MLP over
packed weights holds the gate's and up's whole products and their
`silu_mul` (three 717 MB buffers for 16,384 rows of a 10,944-wide layer);
the Tensor Core split of a long matmul's A holds three bfloat16 terms of
every row (1.08 GB for that layer's down projection); and a `moe-experts`
prefill holds every (token, expert) pair's activation and reads each
expert's weights once per pair. The expanded latent attention's 192-wide
keys and 128-wide values also fell to the untiled attention Kernels.

## What Changes

- **One logits row.** Every prefill graph outside tensor parallelism takes
  the logits row gather a bucket's and the chunk's already take
  (`replay-bucketed-prefill`): the LM head runs on the sampled row only.
- **Row blocks on CUDA.** A long prefill's split Tensor Core matmul serves
  2,048 rows of A per launch, B split once; a gated MLP over packed weights
  holds the gate's and up's products for one block of rows. Every value is
  unchanged.
- **Grouped expert prefill on CUDA.** A `moe-experts` prefill over packed
  `Q8_0` experts groups its pairs by expert and reads an expert's weights
  once per eight of its pairs, 1,024 tokens at a time, each value the
  per-pair Kernels' own.
- **Tiled latent prefill attention on CUDA.** The tiled prefill attention
  Kernel takes a value width of its own: 192-dimension keys with
  128-dimension values.
- **Qualification.** The DeepSeek-V2-Lite long-context step balances the
  layers across both GPUs and prints each GPU's free memory.

## Capabilities

### New Capabilities

- `one-pass-prefill-memory`: what a prefill run in one pass holds beyond its
  rows' activations.

### Modified Capabilities

None.

## Impact

- `magnetar-runtime`: `ComponentGraphCache::prepare` (the gather on every
  prefill graph), the production loop's padding condition.
- `providers/cuda`: `matmul_split_tcp` in row blocks,
  `gated_matmul_split_tcp`, `moe_q8_grouped` and its Kernels,
  `attention_tiled_48_32_kernel`, `CudaMultiProvider::free_memory`.
- `integration-tests/production-loading`: the gains test prints each GPU's
  free memory and names the layer placement.
- `.github/workflows/gpu-runner-smoke.yml`: balanced placement for the
  DeepSeek long-context step, the MoE prefill timing step.
