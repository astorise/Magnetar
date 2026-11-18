## 1. Runtime

- [x] 1.1 `ComponentGraphCache`: chunk length (`prefill_chunk`, `set_prefill_chunk`); `prefill_graph_length` gives the chunk for a longer prompt (not under tensor parallelism); the chunk's graph gets the logits row gather like a bucket's, and its prefill plan serves every longer prompt (`serves_longer_prompts`, `the_chunk_graph_plan_serves_longer_prompts`); `is_bucket_length` keeps pooling and recording to the buckets.
- [x] 1.2 Engine: `execute_chunked_prefill` (`prefill_chunks.rs`): passes of the chunk's graph with the history of the passes before and their first row's position, the last pass padded, the history ending at the prompt; stores sized for the padding.
- [x] 1.3 Production: `MAGNETAR_PREFILL_CHUNK` (2,048; `0` off), `ProductionLoadedModel::set_prefill_chunk` / `prefill_chunk`; no chunking without in-place history; a chunk graph whose attention does not read K and V from its KV history turns chunking off (`prefill_continues_history`).

## 2. Tests

- [x] 2.1 Unit: chunk parsing, passes, stored rows, `prefill_continues_history` (a detached V fails it).
- [x] 2.2 Production on the Reference CPU (`a_production_model_prefills_long_prompts_in_chunks_of_one_graph`): chunks of 4 give the one-pass tokens and first logits (within 1e-5), every long prompt runs the chunk's graph, a prompt of exactly the chunk's length is unpooled.
- [x] 2.3 Tiny Gemma 4 checkpoints (sliding and full layers, shared KV, per-layer inputs, experts) prefilled in chunks of 3 match `transformers` within 1e-3 on the Reference CPU and both CPU Providers (largest difference 1.1e-4); on CUDA too, largest difference 5.2e-5 (run 44251614646).
- [x] 2.4 Tiny DeepSeek-V2 checkpoints (latent attention) decline chunking and still match `transformers`.

## 3. GPU qualification

- [ ] 3.1 `chunked_prefill_matches_one_pass_on_cuda` (`kv` group): Qwen2.5-0.5B Q8_0 (3,000 tokens; chunks of 1,024 and 512) and Mistral 7B Q4_K_M (4,000 tokens; chunks of 2,048 and 1,024): same first token, first-step logits within 0.5, TTFT against one pass.
- [ ] 3.2 TurboQuant gains with chunked prefills: Mistral 7B Q4_K_M at 16,384 and 32,768 tokens in `f32`, `tq4` and `tq3`; Qwen2.5-0.5B TTFT at 16,384 and 32,768 against one pass (run 44203342858).
