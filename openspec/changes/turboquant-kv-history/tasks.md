## 1. Semantics

- [ ] 1.1 `magnetar-provider-api`: codebooks (Lloyd-Max on the sphere-coordinate density, `tq4`/`tq3`), signs, fast Walsh-Hadamard transform, quantize/dequantize of a row of heads, packing.
- [ ] 1.2 Tests: orthogonality and involution of the transform; symmetric, sorted codebooks with the expected MSE at 4 and 3 bits; round trip error within the expected distortion on random rows; packed bytes per row.

## 2. Runtime

- [ ] 2.1 Feature names in `magnetar-core`; `write_zeroed_kv_store` in the Provider API (default unsupported).
- [ ] 2.2 Precision policy (`MAGNETAR_KV_PRECISION`, `_K`, `_V`) resolved per loaded model against advertisements and head widths, with recorded fallback; `KvStoreSpec` and the KV pool carry it.
- [ ] 2.3 Store creation through the Provider call, admission at stored size; KV layout metadata and compatibility.
- [ ] 2.4 Tests with a Provider double: fallback, admission sizes, pooled sets by precision.

## 3. CPU Providers

- [ ] 3.1 `providers/cpu`: TurboQuant stores (f32 storage, rounded rows) and advertisements, both CPU Providers.
- [ ] 3.2 Tiny checkpoints with `tq4`: generation runs and the logit error against `f32` is reported.

## 4. CUDA

- [ ] 4.1 Store, quantizing `kv-write`, download/expansion/copy.
- [ ] 4.2 Decode attention on codes (split-KV, grouped-query); prefill through dequantized scratch.
- [ ] 4.3 Conformance against the Reference CPU on the same rounded stores; step capture.

## 5. Qualification

- [ ] 5.1 Campaigns on Qwen2.5-0.5B, Llama 3.2 1B, Mistral 7B and Gemma 4 E2B: `f32`, `tq4`, `tq3`; greedy agreement, logit error, bytes per token, longest context, TPOT, TTFT.
- [ ] 5.2 Evidence recorded here; default unchanged unless every profile passes.
