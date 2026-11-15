## 1. Runtime: bucketed prefill

- [x] 1.1 Prompt buckets (`MAGNETAR_PREFILL_BUCKETS`), right padding, committed length `n`, and an exact-length fallback above the largest bucket.
- [x] 1.2 `input.logits_row` and the Runtime-inserted `embedding` row gather before the LM head; logits from row `n - 1`.
- [x] 1.3 Tests (`first_native_runtime::prefill_buckets::tests`):
  - bucketed against unpadded prefill on the Reference CPU (tokens identical, logits within tolerance);
  - the first decode step after a bucketed prefill overwrites the padding rows.

## 2. Runtime: KV store pool

- [x] 2.1 Capacity buckets, a per-instance pool keyed by capacity bucket, and Memory Manager accounting under a pool owner.
- [x] 2.2 Bounds (`MAGNETAR_KV_POOL_BYTES`, sets per bucket), LRU eviction, and reclaim before a refused admission.
- [x] 2.3 Tests (`first_native_runtime::kv_pool::tests`): same addresses on reuse, row count reset, one live request per set, reclaim under pressure.

## 3. Prefill capture and replay

- [ ] 3.1 Recording key and table, the per-node / capture / replay sequence, and fallback on capture failure or invalidation.
- [ ] 3.2 CUDA Provider: run the bucketed prefill under capture and fix every Kernel that allocates, synchronizes or downloads during capture.
- [ ] 3.3 Tests:
  - Runtime, with the test step-capture Provider: third request replays; an invalidated recording falls back;
  - CUDA, on real hardware: a captured bucketed prefill replays with identical logits.

## 4. Controls, qualification and documentation

- [ ] 4.1 `set_prefill_replay`, `MAGNETAR_PREFILL_REPLAY=0`, phase ledger `prefill_replay`, and reporting of the pool's held bytes.
- [ ] 4.2 GPU campaign: warm TTFT for short prompts (5, 12 and 40 tokens) on Qwen2.5-0.5B and Mistral 7B, replay against per node in the same run, with token and logit agreement and pool VRAM. Record the evidence in design.md.
- [ ] 4.3 Run cargo fmt, clippy and tests in the affected workspaces and in the CUDA Provider.
