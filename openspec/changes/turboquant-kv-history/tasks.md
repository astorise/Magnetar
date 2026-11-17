## 1. Semantics

- [x] 1.1 `magnetar-provider-api`: codebooks (Lloyd-Max on the sphere-coordinate density, `tq4`/`tq3`), signs, fast Walsh-Hadamard transform, quantize/dequantize of a row of heads, packing.
- [x] 1.2 Tests: orthogonality and involution of the transform; symmetric, sorted codebooks with the expected MSE at 4 and 3 bits; round trip error within the expected distortion on random rows; packed bytes per row.
- [x] 1.3 Store offsets (`turboquant_store_offsets`, `TurboQuantStoreState`): set by the write at row 0, rows quantized as their difference from them; test: a shared component is kept exactly.

## 2. Runtime

- [x] 2.1 Feature names in `magnetar-core`; `write_zeroed_kv_store` in the Provider API (default unsupported).
- [x] 2.2 Precision policy (`MAGNETAR_KV_PRECISION`, `_K`, `_V`) resolved per loaded model against advertisements and head widths, with recorded fallback; `KvStoreSpec` and the KV pool carry it.
- [x] 2.3 Store creation through the Provider call, admission at stored size.
- [x] 2.4 Fallback to f32 for unsupported head widths, with the recorded reason (Runtime test); generation on the Reference CPU with `tq4`/`tq3` stores (tiny Gemma 4 checkpoints).
- [x] 2.5 KV layout metadata and compatibility record the precisions; continuing a history in other precisions is refused (`a_kv_history_is_only_continued_in_its_recorded_precisions`); pooled sets are reused only in their own precisions.

## 3. CPU Providers

- [x] 3.1 `providers/cpu`: TurboQuant stores (f32 storage, rounded rows, offsets) and advertisements, both CPU Providers.
- [x] 3.2 Tiny checkpoints with `tq4`: generation runs and the logit error against `f32` is reported.

## 4. CUDA

- [x] 4.1 Store, quantizing `kv-write`, download/expansion/copy.
- [x] 4.2 Decode attention on codes (split-KV, grouped-query); prefill through dequantized scratch.
- [x] 4.4 Offsets: set by the write at row 0 (`tq_offsets_kernel`, padding rows of a replayed prefill excluded), subtracted before quantizing, added back on read; `q · μ_k` in decode scores, `μ_v` after the inverse rotation (conformance passes on the GPU runner, run 44183609847).
- [x] 4.3 Conformance against the Reference CPU on the same rounded stores, with offsets, 64- to 512-wide heads and a sliding window: passes on the GPU runner (runs 44145395903 and 44183609847) once the test rows are hashed; the smoke `kv` group's generations run with step capture.

## 5. Qualification

- [ ] 5.1 Gains on Qwen2.5-0.5B, Gemma 4 E2B and Mistral 7B (`turboquant-kv-gains.yml`, test `turboquant_kv_gains_on_cuda`): `f32`, `tq4`, `tq3` at 512 to 32,768 tokens; KV bytes per token and tokens held by the free VRAM, which contexts run, TTFT, TPOT, leading tokens and first-step logits against `f32`.
- [x] 5.2 The smoke workflow's `kv` group keeps `tq4` answering "Paris" where `f32` does: it failed for Qwen2.5-0.5B before the offsets (run 44128259285) and passes with them (run 44138881338), with `tq3` too.
- [ ] 5.3 Evidence recorded here; default unchanged unless every profile passes.
