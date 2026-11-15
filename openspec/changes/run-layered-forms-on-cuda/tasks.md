## 1. CUDA Provider

- [x] 1.1 `score_scale` on every attention Kernel; `attention_scaled`.
- [x] 1.2 In-place embedding scale; per-head and weightless RMSNorm; partial RoPE; GELU-gated experts and per-expert scales; softcap, scale, `gelu`, scalar `mul` Kernels.
- [x] 1.3 Executor attributes and advertised features.

## 2. Tests

- [x] 2.1 Conformance against the Reference CPU and `operator_semantics` for every form and attention Kernel path.
- [x] 2.2 A hardware dispatch of every form through the Kernel Registry and executor.
- [ ] 2.3 The GPU runner passes them (astorise/Magnetar-provider-CUDA#42).
