## 1. Runtime

- [ ] 1.1 KV precision policy: instance setting, request override, `MAGNETAR_KV_PRECISION` (K and V), resolution against Provider advertisements with recorded fallback, `f32` default.
- [ ] 1.2 Memory Manager admission at the Provider-reported stored size; conformance test that admitted and allocated sizes match.
- [ ] 1.3 Precision in KV layout metadata and the compatibility key; prefix reuse and resumed sessions reject a precision mismatch.
- [ ] 1.4 Report the precision, bytes per token and longest admitted context in qualification outputs.

## 2. CUDA Provider: bf16

- [ ] 2.1 `bf16` stores: `kv-write` rounding, capture-safe at the advancing offset; download/expand to f32.
- [ ] 2.2 Attention variants (split-KV decode, grouped-query 64/128, block, tiled, Tensor Core prefill) reading `bf16` K and V into their shared tiles.
- [ ] 2.3 Conformance against the Reference CPU on the dequantized history, 1-4,096 keys, causal and windowed, decode and prefill, captured and uncaptured.

## 3. CUDA Provider: q8_0 (then q4_0)

- [ ] 3.1 `q8_0` stores (payload and aligned scale arrays), quantizing `kv-write`.
- [ ] 3.2 Dequantizing attention variants; measure an int8 `q . k` (`__dp4a`) decode variant and keep it only if faster within thresholds.
- [ ] 3.3 Conformance as in 2.3; K and V precisions independent.
- [ ] 3.4 `q4_0` V (and optionally K) after the `q8_0` campaign.

## 4. Qualification

- [ ] 4.1 Campaigns at 512/4,096/16,384 tokens for Qwen2.5-0.5B/1.5B, Llama 3.2 1B, Mistral 7B: logit error, greedy agreement over 128 tokens, perplexity proxy, VRAM per token, longest admitted context on 12 GB, TPOT, TTFT, per precision against `f32`.
- [ ] 4.2 Declare thresholds; promote a default per Provider only when every profile passes; record evidence in this design.
- [ ] 4.3 Tensor-parallel and layer-segment runs with quantized KV (Mistral 7B on two GPUs).

## 5. Documentation

- [ ] 5.1 Document the precisions, controls, memory formulas, measured quality per model and the default.
- [x] 5.2 Point `quantize-and-tier-kv-storage` D08 at this change.
