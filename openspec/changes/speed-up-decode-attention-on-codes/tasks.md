## 1. Staged codes

- [x] 1.1 `tq_attention_split_staged_kernel`: the chunk's codes and norms staged in shared memory (decision 1), every thread in the value sums (decision 2), the queries in dynamic shared memory sized to the group (decision 3).
- [x] 1.2 Conformance unchanged (`turboquant_kv_stores_round_trip_and_attend_like_the_cpu`, `turboquant_exact_heads_attend_like_the_cpu`, captured steps); `decode_attention_times` against run 44290046381. Run 44299614540: conformance passes; no faster than the unstaged Kernel (design, Evidence).

## 2. Split-K

- [x] 2.1 Several chunks per block with an online softmax; the split count from the multiprocessors and the history (decision 4).
- [x] 2.2 Conformance: histories spanning one to many splits, an exact head inside a split, a sliding window, captured steps replayed past their capture's history.
- [x] 2.3 `decode_attention_times` against the codes' Kernel before this change and f32 (run 44299614540): within about 10% either way, slower up to 4,096 keys; the unstaged Kernels stay the default, so TPOT is unchanged.

## 3. Register tiling

- [x] 3.1 The f32 grouped decode Kernel with several multiply-adds per shared load (decision 5); conformance; `decode_attention_times` (run 44302848278): 11-12% faster for 64-wide heads (f32 TPOT -6% to -7% at 16,384 and 32,768 tokens on Qwen2.5-0.5B), unchanged for 128-wide.
- [x] 3.2 The same on the grouped code Kernel (providers/cuda de9dffc): 22% faster for Qwen2.5-0.5B, 29-33% for Mistral 7B, at 4,096 to 32,768 keys (run 44312374879).
- [x] 3.3 TPOT at 512 to 32,768 tokens with both tiled Kernels (run 44314151729): Mistral 7B `tq4` 1.37x f32 at 16,384 tokens, 20% faster than before at 32,768; Qwen2.5-0.5B `tq4` 1.10x f32 at 32,768; outputs unchanged.
- [ ] 3.4 The exact head's chunks in the code Kernel: up to 8% of TPOT at 4,096 tokens.
