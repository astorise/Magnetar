## 1. Operators

- [x] 1.1 `magnetar-core` / `magnetar-provider-api`: `batched-matmul` semantics (`group_count`, `batched_matmul`) and `attention.value_head_stride` (`value_heads`, feature `attention-value-head-stride`).
- [x] 1.2 Reference CPU (runtime): both, with tests (groups against per-group matmuls; strided values; `absorbed_latent_attention_matches_the_expanded_form`: the absorbed decode equals the expanded attention).
- [x] 1.3 `providers/cpu`: both, both CPU Providers (the optimized one leaves strided values to the Reference CPU kernel), advertised.
- [x] 1.4 `providers/cuda`: both (values gathered into a per-shape scratch buffer, then the existing attention), conformance against the CPU (`batched_matmul_and_strided_values_match_the_cpu`, both `batched-matmul` forms, strided values over f32 and tq4 latent stores) passes on the GPU runner (run 44183609847).

## 2. Weights

- [x] 2.1 Hugging Face loader: `k_absorb`, `v_absorb` from `kv_b_proj` (row selections, `[out, in]`).
- [x] 2.2 GGUF loader: the same from `attn_kv_b` and from `attn_k_b` / `attn_v_b` (heads transposed back as for `k_nope_proj`, `f32` from a block-quantized `attn_k_b`).
- [x] 2.3 Runtime: expected shapes of the absorbed tables; the tiny DeepSeek-V2 checkpoints still match `transformers` with them loaded.

## 3. Runtime

- [x] 3.1 Latent layers: K rows `L + r`, no V store (`kv_value_row_widths` 0, `latent_history_width`); the KV state names the K store for both roles; bytes per token, pool and layout metadata follow; `batched-matmul` takes device-resident inputs.

## 4. Component

- [x] 4.1 `components/deepseek` `678d2f6`: expanded prefill writing the latent rows, absorbed decode; rebuilt (two builds give `sha256:59a2f1e8…`), digest repinned.

## 5. Acceptance

- [x] 5.1 The tiny DeepSeek-V2 checkpoints (Hugging Face and GGUF) against `transformers`: they match on the Reference CPU, both CPU Providers and with every value device-resident (largest logit difference 1.2e-5); on CUDA too, largest logit difference 1.3e-5 (run 44183609847).
- [x] 5.2 DeepSeek-V2-Lite-Chat Q8_0 on two GPUs continues "The capital of France is" with " Paris." (run 44183609847). Its history is 576 values per token and layer against 5,120. TPOT after a short prompt: 20.30 ms against 12.25 ms with the expanded history (run of 11:12 UTC the same day): slower, see 5.3.

- [ ] 5.3 The absorbed decode's TPOT at least the expanded form's after a short prompt. Run 44195880740: 13.02 ms with the f32 latent history and 13.45 ms with the tq4 one (was 20.30 ms), against 12.25 ms expanded. The decode steps are captured and replayed (`[step capture] captured: decode step 2`); 7.3 ms of the difference came from the transposed `batched-matmul` reading one weight row per thread, which a warp per output now reads coalesced. Of the 13 ms per step, replay takes 7.1 ms and the logits' materialization 4.9 ms (for both histories).

## 6. TurboQuant on the latent history (`turboquant-kv-history`, second step)

- [x] 6.1 A latent history is quantized in groups of gcd(`kv_lora_rank`, `qk_rope_head_dim`) values (64 in DeepSeek-V2 and V3: nine groups per 576-value row, 324 bytes per row and layer in `tq4` against 2,304 in f32); narrower groups keep f32 with the reason recorded.
- [x] 6.2 CUDA: strided attention values read from a TurboQuant store through dequantized scratch, the store read once for K and V; conformance on a tq4 store of 576-value rows.
- [x] 6.3 The tiny GGUF checkpoints (the `wide` bundle) run with `tq4` and `tq3` latent histories on the Reference CPU, and both CPU Providers match it exactly; the Hugging Face ones keep f32 (groups of 4).
- [x] 6.4 DeepSeek-V2-Lite-Chat with a `tq4` latent history continues with " Paris.": its 64 greedy tokens equal the f32 run's, TPOT 20.49 ms (run 44183609847).
- [x] 6.5 CUDA: a decode step over a `tq4`/`tq3` latent history runs on its codes (`tq_latent_attention_split_kernel`), and decode steps read the values in place: 921.5 against 8,105.5 us through dequantized scratch at 32,768 keys (design, Evidence: latent decode on codes).
- [x] 6.6 The f32 latent decode reads the history once per query head (6,886.6 us at 32,768 keys): a block serving eight query heads (`latent_attention_split_kernel`), 1,499.6 us (run 44355981594).
- [ ] 6.7 DeepSeek-V2-Lite's TPOT at long contexts with f32 and `tq4` latent histories. Run 44358602056: at 1,024 tokens 15.06 ms (f32), 15.31 ms (`tq4`), 17.96 ms (`tq4` with the exact head, whose keys the code Kernel scores against queries read from global memory); from 4,096 tokens the one-pass prefill runs out of memory on two 12 GB GPUs. The exact keys now read as float4s: a 4,096-key decode step with the head 227.7 against 295.0 us (run 44369519228). With the layers balanced across the GPUs (run 44374637064), 4,096 tokens run: TPOT 16.56 ms (f32), 16.97 ms (`tq4`, 0.98x), 17.96 ms (`tq4` with the exact head), 64/64 tokens equal to f32; 16,384 tokens run out of the first GPU's memory in layer 0's gated MLP (`bound-one-pass-prefill-memory`).
