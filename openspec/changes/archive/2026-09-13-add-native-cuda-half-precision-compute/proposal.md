## Why

The README's remaining scope explicitly names this gap: "`F16`/`BF16` weight storage is decoded and converted to `F32` at Model Loading time, but native CUDA `F16`/`BF16` compute kernels do not exist yet." Closing it for real turned out to be architecturally deeper than any prior chantier this session: `providers/cuda`'s `CudaDeviceBuffer` is hardcoded `CudaSlice<f32>` throughout the whole crate, and `magnetar-runtime`'s `HostTensor` (the Provider-agnostic host/device transport type every Provider depends on) is hardcoded `Vec<f32>`. A real, honest native-half-precision-compute capability is a multi-crate, multi-phase effort, not a single bounded increment -- this proposal's own `design.md` lays out the full plan across `magnetar-runtime` and `providers/cuda`, but this specific change's scope is Phase 1 only: the `magnetar-runtime`-side conversion primitives every later phase depends on.

## What Changes

- `magnetar-runtime` gains real `f32_to_f16`/`f32_to_bf16` encoders, completing the `f16_to_f32`/`bf16_to_f32` decoders `model_loading.rs` already has and already uses to convert stored half-precision weights to `F32` at load time. Today only the decode direction exists because nothing in this codebase has ever needed to encode `F32` back down to a half-precision bit pattern; a real device-side half-precision buffer (Phase 2, out of scope here) needs this to convert host `F32` values into on-device `F16`/`BF16` bytes at the upload boundary, and back on download.
- `HostTensor` itself is **not** retyped in this change (see `design.md`'s "Decisions" for why: it stays `Vec<f32>`, exactly as today). No behavior of any existing Provider, Kernel, or Model Loading path changes.
- **BREAKING**: none. This change adds two new private-to-crate functions and their tests; nothing existing is touched.

## Capabilities

### New Capabilities
(none -- this is an internal conversion-primitive addition; the "Native CUDA Half-Precision Compute" capability itself is introduced by Phase 2, not this change)

### Modified Capabilities
- `model-loading`: extends the existing "DType Handling" requirement with a scenario covering the new encode direction (`F32` -> `F16`/`bfloat16`), alongside its pre-existing decode-direction behavior.

## Impact

- `magnetar-runtime/src/model_loading.rs`: two new functions, `f32_to_f16`/`f32_to_bf16`, placed directly beside their existing decoder counterparts.
- Tests: exhaustive round-trip verification -- for all 65536 possible `u16` bit patterns interpreted as `f16` (respectively `bf16`), decoding via the already-trusted `f16_to_f32`/`bf16_to_f32` and re-encoding via the new functions reproduces the original bit pattern exactly (NaN payloads excepted, where "decodes back to a NaN" is checked instead of exact bit equality) -- far stronger evidence than hand-picked test cases for numerically tricky bit-manipulation code, following this session's established "copy/verify the tested primitive, don't just hand-check a few cases" practice for half-precision conversion.
- No other crate is touched. `design.md` documents Phase 2 (`providers/cuda`: dtype-tagged `CudaDeviceBuffer`, boundary conversion at upload/download, a real native half-precision elementwise kernel, verified on real hardware) and Phase 3 (Runtime-side compute-dtype negotiation) as explicit, deliberately deferred future chantiers -- not attempted here.
