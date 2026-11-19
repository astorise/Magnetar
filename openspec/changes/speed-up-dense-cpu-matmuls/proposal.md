## Why

The CPU benchmark cell measured one request of 5 prompt tokens and 16 greedy new tokens on Qwen2.5-0.5B-Instruct, on a 4-vCPU AMD EPYC 9V45 runner:

| Engine | Request | Decode, per token |
| --- | --- | --- |
| Magnetar, Reference CPU Provider (run 44485741553) | 19.9 s | about 1.2 s |
| Magnetar, optimized CPU Provider (run 44511061319) | 7.49 s | 419 ms |
| llama-server, FP16 GGUF (run 44485741553) | 0.357 s | 21.7 ms |

The optimized CPU Provider (`execute-quantized-weights-on-cpu`) computes from packed, block-quantized weights. Every other Kernel runs on the Reference CPU kernel it wraps. The weights of a bfloat16 or f32 checkpoint are not packed, so every matmul of such a model ran on one thread without vector code, at 4.8 GB/s of f32 weights.

## What Changes

- **Compact bfloat16 weights.** A Model Instance weight whose every value is exactly a bfloat16 value, as every weight of a bfloat16 checkpoint is, is stored as its bfloat16 bits through `write_weight_tensor`. The CUDA Provider already does this (`execute-native-half-precision`). It halves the bytes and keeps every value unchanged. `MAGNETAR_CPU_COMPACT_WEIGHTS=0` keeps such weights in f32.
- **Dense matmuls.**
  - `matmul`, `matmul-bias` and `matmul-residual` against a dense weight, compact or f32, run across the Provider's pool, by output column.
  - They use AVX2/FMA kernels for a weight used transposed (output-major) or not (input-major), with a scalar fallback.
  - A transposed `a`, or shapes the kernel does not take, still run on the Reference CPU kernel, which reports any error.
- **Lookups.** The embedding lookup gathers a compact table's rows in place.
- **The benchmark's CPU cell.** The qualification campaign accepts `cpu-native` in `MAGNETAR_QUALIFICATION_PROVIDERS`, and the benchmark's CPU cell selects it instead of the Reference CPU Provider.

## Capabilities

### New Capabilities

- `dense-cpu-matmuls`: compact bfloat16 weights on the optimized CPU Provider, pooled vector matmuls against dense weights, in-place compact lookups, and the benchmark's CPU Provider.

### Modified Capabilities

None.

## Impact

- `providers/cpu` (`native::dense`, `native::mod`): compact storage, the dense kernels and their dispatch.
- `integration-tests/production-loading`: the `cpu-native` qualification Provider.
- `benchmark/magnetar`: the CPU cell's Provider.
- No Runtime, WIT or Component change.
