## 1. CUDA Provider

- [x] 1.1 `matmul_tcp_split_kernel` (`tcp_matmul<NB, LO_LAST>`), `CudaKernels::matmul_split_tcp`, the long packed matmul path.
- [x] 1.2 Conformance: `long_prefills_against_packed_weights_match_reference_cpu` (Q8_0, Q4_K, Q6_K; 64, 200 and 300 rows; plain, bias, residual) within 1e-5 of the largest output.

## 2. Qualification

- [ ] 2.1 GPU runner: the conformance tests pass on the RTX 3060. `long_prefills_against_packed_weights_match_reference_cpu` passes; `q8_packed_gated_matmul_and_embedding_match_reference_cpu` fails at 70 rows (run 44266581438). See design.md, Open question.
- [x] 2.2 TTFT (run 44266581438 against 44256946982): Mistral 7B Q4_K_M 3.05x faster at 4,096 tokens (40.0 to 13.1 s), 2.72x at 16,384, 2.37x at 32,768; Qwen2.5-0.5B Q8_0 2.10x at 4,096, 1.56x at 32,768. Generated tokens unchanged. See design.md, Evidence.
