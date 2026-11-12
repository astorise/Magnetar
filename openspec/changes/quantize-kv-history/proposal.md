## Why

Weights now run packed (`execute-direct-quantized-weights`): Mistral 7B needs 2.0-7.7 GB of VRAM for its weights depending on the export. The KV history has not changed: every layer's in-place K and V stores (`append-kv-history-in-place`) are f32 rows of `kv_heads x head_dim` values. For Mistral 7B (32 layers, 8 KV heads of 128) that is 256 KiB per token: 1 GiB at 4,096 tokens, 8 GiB at 32,768, more than its `Q4_K_M` weights from about 17,000 tokens on, per sequence. On a 12 GB RTX 3060 the KV history, not the weights, now bounds context length and the number of concurrent sequences, and decode attention reads all of it every step (Qwen2.5-0.5B TPOT grows from 4.4 to 5.1 ms between 512 and 4,096 tokens of context).

This is D08 of the roadmap (`quantize-and-tier-kv-storage`, which keeps D09, KV tier movement); this change makes D08 concrete for the CUDA Provider first.

## What Changes

- A **KV storage precision** per Model Instance, chosen by the Runtime from what the selected Provider advertises and the request or instance policy: `f32` (today, the reference), `bf16`, `q8_0`, and later `q4_0`. K and V may differ (for example K `q8_0`, V `q4_0`), as quality studies favor keeping K finer.
- The **KV store resources keep their logical f32 contract**: `kv-write` still takes f32 rows and `attention` still sees f32 values. The Provider stores and reads them in the chosen precision; nothing in the WIT graph contract or the Components changes.
- **Memory Manager admission at the stored size**: a store's admitted bytes follow its precision (`f32` 4, `bf16` 2, `q8_0` 34/32, `q4_0` 18/32 bytes per value), so the savings become capacity (longer contexts, more sequences), not only unused VRAM.
- **CUDA kernels**:
  - `kv-write` quantizes the new rows into the store (per 32-value block within a row: `d = max|x| / 127`, round to nearest even, as for int8 activations; `bf16` round to nearest even);
  - every attention path (split-KV decode, grouped-query decode for 64- and 128-dim heads, block, tiled prefill, Tensor Core prefill) dequantizes K and V while staging them into shared memory, so global reads shrink by the precision ratio;
  - decode attention over quantized K may multiply int8 queries with int8 keys through `__dp4a` (queries quantized per block), measured against dequantize-then-f32.
- **Compatibility**: KV precision joins the KV compatibility key (`kv-cache` "KV Cache Compatibility"), so a history written in one precision is never read as another; prefix reuse, rollback (`append-kv-history-in-place`) and tensor-parallel replication keep working per precision.
- **Controls**: an instance setting and request override, an environment variable mirroring it for CI (`MAGNETAR_KV_PRECISION=f32|bf16|q8_0|...`), `f32` the default until the qualification below promotes `bf16` or `q8_0`.
- **Qualification**: long-context campaigns (512, 4,096, 16,384 tokens where they fit) on Qwen2.5 and Mistral 7B comparing each precision with `f32`: logit error against the f32 history, greedy token agreement, a perplexity proxy on a fixed corpus, VRAM per token, the longest context that fits, TPOT and TTFT.

Status: **Proposed.** No code changes with this proposal.

## Capabilities

### New Capabilities

- `quantized-kv-history`: KV storage precision selection, admission at stored size, Provider storage and attention over quantized history, compatibility, controls and qualification.

### Modified Capabilities

- `kv-cache`: KV precision becomes part of the layout metadata and of the compatibility key (delta in this change).

## Impact

- `magnetar-runtime`: KV precision policy and resolution, Memory Manager admission per precision, compatibility key, controls, campaign reporting.
- `providers/cuda`: quantizing `kv-write`, dequantizing attention variants, capability advertisement; `providers/cpu` and the planned optimized CPU Provider advertise `f32` only until they implement more.
- No WIT change; Components are unaffected.

## Dependencies

- `append-kv-history-in-place` (in-place stores, `key_count`, copy-free commit/rollback).
- `execute-direct-quantized-weights` (int8 activation quantization scheme reused for `q8_0` KV).
- Related: `operate-paged-kv-storage` (D06) can page quantized stores later; `quantize-and-tier-kv-storage` (D09) moves them between tiers.

## Non-goals

- KV tier movement (D09) and paged KV (D06).
- FP8 KV (no native FP8 on the RTX 3060's Ampere); it can follow on hardware that has it.
- Changing the Reference CPU Provider's f32 history.
- Learned or per-channel calibrated KV quantization (KIVI-style asymmetric per-channel K); considered after the symmetric block scheme is measured.
