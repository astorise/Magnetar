## 1. CUDA Provider

- [x] 1.1 `tc_split3_kernel` (three bfloat16 terms), `matmul_tcp_split3_kernel` (six products, a partial per tile added in f32, dynamic shared memory), `CudaKernels::split_bf16_3_all` and `matmul_split_tcp`, the long packed matmul path.
- [x] 1.2 Conformance: `long_prefills_against_packed_weights_match_reference_cpu` (Q8_0, Q4_K, Q6_K; 64, 200 and 300 rows; plain, bias, residual) within 1e-5 of the largest output.

## 2. Qualification

- [x] 2.1 GPU runner: every conformance test passes at its unchanged tolerance (run 44283381146, the gated packed `Q8_0` case at 70 rows included, which failed with two terms and three products in run 44266581438 and with six products in one chain in run 44274927793). TTFT 1.75x faster than the f32 path for Mistral 7B Q4_K_M at 4,096 tokens (40.0 to 22.9 s), 1.57x at 32,768; 1.19x to 1.30x for Qwen2.5-0.5B Q8_0; generated tokens unchanged (design, Evidence).
- [x] 2.2 Two terms, three products (run 44266581438 against 44256946982): Mistral 7B Q4_K_M 3.05x faster at 4,096 tokens (40.0 to 13.1 s), 2.72x at 16,384, 2.37x at 32,768; Qwen2.5-0.5B Q8_0 2.10x at 4,096, 1.56x at 32,768; generated tokens unchanged. Below f32 precision: replaced (design, decision 2).
