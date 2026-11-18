## 1. CUDA Provider

- [x] 1.1 `matmul_tcp_split_kernel` (`tcp_matmul<NB, LO_LAST>`), `CudaKernels::matmul_split_tcp`, the long packed matmul path.
- [x] 1.2 Conformance: `long_prefills_against_packed_weights_match_reference_cpu` (Q8_0, Q4_K, Q6_K; 64, 200 and 300 rows; plain, bias, residual) within 1e-5 of the largest output.

## 2. Qualification

- [ ] 2.1 GPU runner: the conformance test passes on the RTX 3060.
- [ ] 2.2 TTFT of Mistral 7B Q4_K_M at 4,096, 16,384 and 32,768 tokens and of Qwen2.5-0.5B Q8_0 at 4,096 to 32,768, against run 44256946982; generation unchanged against f32 history (`kv` group).
