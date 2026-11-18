## 1. Staged codes

- [ ] 1.1 `tq_attention_split_grouped_kernel`: the chunk's codes and norms staged in shared memory (decision 1), every thread in the value sums (decision 2), the queries in dynamic shared memory sized to the group (decision 3).
- [ ] 1.2 Conformance unchanged (`turboquant_kv_stores_round_trip_and_attend_like_the_cpu`, `turboquant_exact_heads_attend_like_the_cpu`, captured steps); `decode_attention_times` against run 44290046381.

## 2. Split-K

- [ ] 2.1 Several chunks per block with an online softmax; the split count from the multiprocessors and the history (decision 4).
- [ ] 2.2 Conformance: histories spanning one to many splits, an exact head inside a split, a sliding window, captured steps replayed past their capture's history.
- [ ] 2.3 `decode_attention_times` and TPOT at 4,096 to 32,768 tokens (Qwen2.5-0.5B, Mistral 7B) against the codes' Kernel before this change and f32.
