## 1. Semantics

- [x] 1.1 `magnetar-provider-api`: `TurboQuantStoreState` with an exact head (rows below `E` unchanged), stored sizes with the head, tests.
- [x] 1.2 The store creation call carries the head (`write_zeroed_kv_store_with_exact_head`, advertised by `turboquant-kv-exact-head` on `kv-write`); the Reference CPU and both CPU Providers honour it.

## 2. Runtime

- [x] 2.1 `MAGNETAR_KV_EXACT_ROWS` / `KvPrecisions::with_exact_rows`, resolution with the precisions, admission at the head's size, KV compatibility and layout metadata.
- [x] 2.2 Tests: a history shorter than `E` matches f32 stores exactly; a longer one matches the round trip of its rows from `E` on; histories are only continued with the same head. (`an_exact_head_keeps_its_rows_unchanged`; the tiny Gemma 4 and DeepSeek-V2 checkpoints' `..._keep_an_exact_..._head`, bit-identical to f32 under the default head; `kv_quantization_modes_name_turboquant_precisions_only` for the compatibility key. The codes-only tests now ask for `with_exact_rows(0)`.)

## 3. CUDA

- [x] 3.1 The head buffer; `tq_kv_write_kernel` copies the head's rows into it.
- [x] 3.2 Decode by regime with the capture bound; past the heads, the code split Kernels read the heads' keys (design decision 4); prefill on the heads or through a scratch that reads them.
- [x] 3.3 Conformance against the CPU semantics: histories shorter and longer than `E`, grouped and per-head decode, captured steps crossing `E` (`turboquant_exact_heads_attend_like_the_cpu`, `captured_decode_steps_keep_their_exact_head_regime`; passed on the RTX 3060 in runs 44289082020 and 44290046381).

## 4. Qualification

- [ ] 4.1 TPOT against f32 at 512 to 32,768 tokens (Qwen2.5-0.5B, Mistral 7B): at least f32's at 512 tokens; the codes-only decode's from 4,096 on. Run 44289082020: f32's at 512 tokens (1.01x and 0.99x) and `tq4` within 3% of the codes-only decode past the head; `tq3` past the head 8-24% slower than without a head in the previous run (suspected accumulators in local memory, removed in providers/cuda 5164290, to be measured).
- [x] 4.2 Agreement with f32 (leading tokens, first-step logits) against the codes-only history (run 44289082020; design, Evidence): f32's logits exactly up to the head; past it, closer to f32 on Mistral 7B, as close on Qwen2.5-0.5B.
