## 1. Operators and Reference CPU

- [x] 1.1 Add `kv-write` to the initial operator catalog (in-place memory behavior, `row_offset` attribute, capacity check) and the optional `attention` `key_count` attribute.
- [x] 1.2 Reference CPU: `kv-write` Kernel; `attention` honours `key_count` and advertises `key-count`. Unit tests, including a NaN-filled tail.

## 2. First-native engine

- [x] 2.1 Decide the in-place path once per request (Provider capabilities, whole-model step, capacity); keep the concat path otherwise.
- [x] 2.2 Prefill and decode write rows into per-layer stores; attention receives `key_count`; commit advances committed lengths without copies; discard and release free stores.
- [x] 2.3 Tests: in-place and concat paths produce bit-identical logits on the canonical fixture; failed or discarded steps leave committed rows untouched; resident-model bookkeeping stays bounded.

## 3. Providers

- [x] 3.1 `providers/cpu`: `kv-write` Kernel and `key_count` in attention, with the `key-count` feature.
- [x] 3.2 `providers/cuda`: in-place `kv-write`; both attention kernels honour `key_count`; device conformance with NaN-filled tails.

## 4. Qualification

- [ ] 4.1 Real Qwen2.5-0.5B on Reference CPU and CUDA: logits unchanged against the independent corpus.
- [ ] 4.2 GPU CI: record KV append, commit, decode step and TPOT before and after.
