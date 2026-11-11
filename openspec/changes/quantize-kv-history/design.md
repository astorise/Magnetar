## Context

In-place KV history: per layer, one K and one V store resource of `[capacity, kv_heads * head_dim]` f32 rows; `kv-write` appends the step's rows at `row_offset`, attention reads the first `key_count` rows (`append-kv-history-in-place`). CUDA keeps both as f32 `CudaDeviceBuffer`s, which already support a compact representation for weights (bf16, packed quantized formats).

KV size per token, f32: `2 x layers x kv_heads x head_dim x 4` bytes.

| Model | Per token f32 | 4,096 tokens | 32,768 tokens |
|---|---|---|---|
| Qwen2.5-0.5B (24 x 2 x 64) | 24 KiB | 96 MiB | 768 MiB |
| Llama 3.2 1B (16 x 8 x 64) | 64 KiB | 256 MiB | 2 GiB |
| Mistral 7B (32 x 8 x 128) | 256 KiB | 1 GiB | 8 GiB |

| Precision | Bytes per value | Ratio to f32 |
|---|---|---|
| `f32` | 4 | 1 |
| `bf16` | 2 | 2.0x smaller |
| `q8_0` | 34/32 | 3.8x |
| `q4_0` | 18/32 | 7.1x |

## Goals / Non-Goals

**Goals:** cut KV memory 2-7x with measured, bounded quality loss; turn the savings into admitted capacity; speed up long-context decode attention by reading fewer bytes; keep the logical f32 contract so Components and the WIT graph contract are untouched.

**Non-Goals:** see the proposal.

## Decisions

### 1. Precision is a Runtime policy, storage is the Provider's

The Runtime decides the precision per Model Instance (instance setting, request override, environment control) among those the selected Provider advertises for both `kv-write` and every attention variant the plan uses; if a precision is not advertised for one of them, the plan falls back to the next advertised precision, ending at `f32`, and records why. The store resources keep their logical f32 descriptor for the graph; their admitted size uses the stored precision. The Provider allocates the store in that precision and owns its layout.

Alternative rejected: a new graph-visible dtype on the KV resources, which would change the WIT graph contract and every Component for what is a storage choice.

### 2. Block layout

`q8_0` / `q4_0` blocks of 32 values along a row, never crossing a head (`head_dim` is a multiple of 32 for every supported model), each block a signed 8-bit or 4-bit payload and one f16 scale, rows contiguous so `row_offset` and `key_count` keep their meaning. The scales live in a separate array aligned for vector loads (as for packed weights on CUDA), so a K chunk loads as one payload span and one scale span. `bf16` stores rounded halves.

### 3. Writing

`kv-write` quantizes the incoming f32 rows in the same kernel that writes them (one warp per block, as `quantize_activations_q8_kernel`), at the captured step's advancing offset, so step capture keeps one launch per store and allocates nothing. Rollback stays copy-free: rows past the committed count are overwritten later, whatever their precision.

### 4. Reading

Every attention kernel already stages K and V chunks into shared memory; the quantized variants load compact payloads and scales and dequantize into the same shared tiles, leaving the softmax and accumulation code unchanged (f32). Variants: split-KV decode, grouped-query decode (64- and 128-dim heads), block and tiled prefill, Tensor Core prefill (dequantize into the bf16 tiles it already uses). A second step evaluates int8 `q . k` with `__dp4a` against quantized queries for `q8_0` K in decode, kept only if faster and within the quality bounds.

### 5. Compatibility and multi-GPU

The precision pair (K, V) joins the KV compatibility key and the layout metadata: prefix reuse and resumed sessions require equal precision, else they re-prefill. Tensor parallelism replicates attention and its stores per member, each in the same precision; layer segments keep each store on its layer's Device.

### 6. Qualification

Campaigns at 512, 4,096 and 16,384 tokens (where they fit) for Qwen2.5-0.5B, Qwen2.5-1.5B, Llama 3.2 1B and Mistral 7B, each precision against `f32` on the same Provider: max and mean logit error at every generated position, greedy token agreement over 128 tokens, a perplexity proxy (teacher-forced negative log-likelihood on a fixed text) whose relative change has a declared threshold, VRAM per token, the longest context admitted on a 12 GB Device, TPOT and TTFT. Promotion of a default requires all thresholds on every profile.

## Risks / Trade-offs

- [Quality loss grows with context and is model-dependent (outlier channels in K)] -> K and V precisions independent; thresholds per profile; `q4_0` K not proposed as a default.
- [Dequantization in attention costs compute] -> measured per variant; at long context the kernels are bandwidth-bound, so fewer bytes should win; short contexts may keep `bf16`.
- [Non-determinism between precisions confuses reuse] -> precision in the compatibility key; never read a store in another precision.
- [Admission reports savings the Provider does not realize] -> admission uses the size the Provider reports for the store it allocated, checked by a conformance test.

## Migration Plan

`f32` stays the default. `bf16` and `q8_0` land behind the control, are qualified, then a default may be promoted per Provider. Rollback is the control set to `f32`; persisted KV snapshots (D09) carry their precision and are re-prefilled when incompatible.

## Open Questions

- Default after qualification: `bf16` (near-lossless) or `q8_0` (3.8x).
- Whether K and V defaults differ (`q8_0` K with `q4_0` V).
- `q4_0` scheme: symmetric blocks or asymmetric per-channel for K.
