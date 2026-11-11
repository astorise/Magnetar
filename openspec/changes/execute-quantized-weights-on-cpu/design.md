## Context

On CUDA, direct quantized execution rests on four pieces: packed bytes kept per format (`CudaQ8Weight`), activations quantized to int8 per 32-value block before each packed matmul, per-format integer dot kernels (`__dp4a`), and gated forms for the MLP. With them every format reaches 180-310 GB/s and decode cost follows the file size. The CPU side has none of these: Model Loading hands the Reference CPU Provider f32 weights, and its kernels are straightforward loops meant to be read, not to be fast.

A CPU decode step is bandwidth-bound in the same way: on a dual-channel DDR5 desktop (about 60-80 GB/s) a 4.4 GB `Q4_K_M` 7B costs at least 55-75 ms per token, against at least 360-480 ms for the 29 GB f32 form. The goal is to approach that bound.

## Goals / Non-Goals

**Goals:** run every packed format on the CPU from its own bytes, near memory bandwidth on decode, with the same numerics as the CUDA int8 path; keep a scalar oracle and disable controls; no new dependency.

**Non-Goals:** see the proposal.

## Decisions

### 1. A separate optimized Provider, not a faster Reference CPU

`magnetar:provider/cpu-native` is a new Provider identity with its own Device (`DeviceType::Cpu`, compute units, detected instruction sets as `magnetar:cpu-feature/*` capabilities, already produced by `detected_simd_capabilities`). It delegates operators it does not optimize to the Reference CPU kernels (the same crate's public functions), so it is complete from the start. The Reference CPU Provider stays the oracle; conformance compares the two.

Alternative rejected: optimizing `providers/cpu` in place, which would break "Reference CPU Provider Is Not Optimized Provider" and lose the independent oracle.

### 2. Storage

Packed weights stay host-resident as their checkpoint blocks, transposed storage as offered (stored row = output column, contiguous `k`), so a column's dot product streams one contiguous run. Per format, the same reorganizations as CUDA where they help alignment (scales split from quants for `Q6_K`, `Q2_K`, `Q3_K`, the i-quants; `Q3_K` scales unpacked), done once at `write_packed_weight_tensor`. The Memory Manager admits the packed size.

### 3. Int8 activations and integer dots

Per matmul, the activation rows are quantized into a reusable per-thread scratch: per 32 values, `d = max|x| / 127`, `q = round_ties_even(x / d)` (the CUDA rule, so the two Providers agree on rounding), plus each block's sum for the formats with mins (`Q4_K`, `Q2_K`). Per format the kernel decodes weight bytes into signed 8-bit lanes and accumulates `sum(q_w * q_a)` in 32-bit integers per block, then scales in f32:

| ISA | 32-value block dot |
|---|---|
| AVX-512 VNNI / AVX-VNNI | `vpdpbusd` on unsigned-offset activations (bias corrected by the block sum) |
| AVX2 | `vpmaddubsw` (unsigned x signed) + `vpmaddwd` |
| AArch64 dotprod | `sdot` |
| NEON | `smull` + `sadalp` |
| scalar | `i32` loop, the oracle |

Unsigned-times-signed instructions take the weights as the signed side; `IQ4` table lookups use `vpshufb`/`tbl`, IQ2/IQ3 grids load 8-byte entries from `iq_grids` and apply signs with a byte mask (`vpsignb`-style or xor/sub), as on CUDA.

Dispatch is chosen once per Provider from `is_x86_feature_detected!` / `is_aarch64_feature_detected!`, capped by `MAGNETAR_CPU_SIMD`.

### 4. Threads

A persistent pool of `min(available_parallelism, MAGNETAR_CPU_THREADS)` workers (std threads, parked between steps) splits output columns in contiguous ranges aligned to cache lines; activations are quantized once by the caller and shared read-only. Decode matmuls are memory-bound, so the pool is sized for bandwidth, typically the physical core count. Other operators run on the calling thread until profiled.

### 5. Longer matmuls

For `m` beyond the row tile (prefill), weights are dequantized one panel (for example 64 output columns) at a time into a per-thread f32 scratch and multiplied with a blocked f32 kernel; nothing larger than a panel is materialized.

### 6. Numerics and tests

- Each SIMD kernel equals the scalar kernel exactly per block (integer sums) and to f32 summation-order tolerance overall; the scalar kernel matches the Reference CPU on the int8-rounded activations, exactly as the CUDA conformance tests do.
- `MAGNETAR_CPU_INT8_ACTIVATIONS=0` keeps f32 activations through the panel-dequantizing path, matching the Reference CPU on the f32 weights.
- Real checkpoints: tokens compared with the CUDA int8 run and with the Reference CPU f32 run (warning on divergence, "Paris" required), as for CUDA.

## Risks / Trade-offs

- [ISA matrix grows test surface] -> one scalar oracle; each SIMD level forced in CI through `MAGNETAR_CPU_SIMD`; AVX-512 and `sdot` only where the CI hardware has them, reported as skipped otherwise.
- [Threads compete with the GPU Provider's host work] -> the pool bound control; the CPU Provider is resolved only when policy or affinity selects it.
- [Int8 activations change outputs] -> same scheme and disable control as CUDA; divergence reported per checkpoint.
- [Delegated f32 operators (attention over long contexts) dominate after matmuls] -> measured per phase (`A06` ledger) before optimizing them.

## Migration Plan

Additive: a new Provider, unselected unless policy resolves it. Rollback is not resolving it or `MAGNETAR_CPU_PACKED_WEIGHTS=0`.

## Open Questions

- Provider identity and packaging: a second crate in `magnetar-provider-cpu` or a feature of the existing one.
- Default thread count: physical cores (needs topology detection) or `available_parallelism`.
- Whether `wasm32` gets `simd128` kernels in this change or a follow-up.
