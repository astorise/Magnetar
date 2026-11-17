## 1. Operators

- [x] 1.1 `magnetar-core` / `magnetar-provider-api`: `batched-matmul` semantics (`group_count`, `batched_matmul`) and `attention.value_head_stride` (`value_heads`, feature `attention-value-head-stride`).
- [x] 1.2 Reference CPU (runtime): both, with tests (groups against per-group matmuls; strided values; `absorbed_latent_attention_matches_the_expanded_form`: the absorbed decode equals the expanded attention).
- [x] 1.3 `providers/cpu`: both, both CPU Providers (the optimized one leaves strided values to the Reference CPU kernel), advertised.
- [ ] 1.4 `providers/cuda`: both (values gathered into a per-shape scratch buffer, then the existing attention), conformance against the CPU written (`batched_matmul_and_strided_values_match_the_cpu`); runs on the GPU runner.

## 2. Weights

- [x] 2.1 Hugging Face loader: `k_absorb`, `v_absorb` from `kv_b_proj` (row selections, `[out, in]`).
- [x] 2.2 GGUF loader: the same from `attn_kv_b` and from `attn_k_b` / `attn_v_b` (heads transposed back as for `k_nope_proj`, `f32` from a block-quantized `attn_k_b`).
- [x] 2.3 Runtime: expected shapes of the absorbed tables; the tiny DeepSeek-V2 checkpoints still match `transformers` with them loaded.

## 3. Runtime

- [x] 3.1 Latent layers: K rows `L + r`, no V store (`kv_value_row_widths` 0, `latent_history_width`); the KV state names the K store for both roles; bytes per token, pool and layout metadata follow; `batched-matmul` takes device-resident inputs.

## 4. Component

- [x] 4.1 `components/deepseek` `678d2f6`: expanded prefill writing the latent rows, absorbed decode; rebuilt (two builds give `sha256:59a2f1e8…`), digest repinned.

## 5. Acceptance

- [ ] 5.1 The tiny DeepSeek-V2 checkpoints (Hugging Face and GGUF) against `transformers`: they match on the Reference CPU, both CPU Providers and with every value device-resident (largest logit difference 1.2e-5); CUDA on the GPU runner.
- [ ] 5.2 DeepSeek-V2-Lite-Chat on CUDA continues "The capital of France is" with "Paris"; bytes per token and TPOT against the expanded history.
