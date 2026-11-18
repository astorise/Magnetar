## 1. Staged codes

- [x] 1.1 `tq_attention_split_staged_kernel`: the chunk's codes and norms staged in shared memory (decision 1), every thread in the value sums (decision 2), the queries in dynamic shared memory sized to the group (decision 3).
- [x] 1.2 Conformance unchanged (`turboquant_kv_stores_round_trip_and_attend_like_the_cpu`, `turboquant_exact_heads_attend_like_the_cpu`, captured steps); `decode_attention_times` against run 44290046381. Run 44299614540: conformance passes; no faster than the unstaged Kernel (design, Evidence).

## 2. Split-K

- [x] 2.1 Several chunks per block with an online softmax; the split count from the multiprocessors and the history (decision 4).
- [x] 2.2 Conformance: histories spanning one to many splits, an exact head inside a split, a sliding window, captured steps replayed past their capture's history.
- [x] 2.3 `decode_attention_times` against the codes' Kernel before this change and f32 (run 44299614540): within about 10% either way, slower up to 4,096 keys; the unstaged Kernels stay the default, so TPOT is unchanged.

## 3. Register tiling

- [ ] 3.1 The f32 grouped decode Kernel with queries and weights held in registers across several keys and coordinates (decision 5); conformance; `decode_attention_times`.
- [ ] 3.2 The same on the code Kernels if 3.1 confirms the bound.
