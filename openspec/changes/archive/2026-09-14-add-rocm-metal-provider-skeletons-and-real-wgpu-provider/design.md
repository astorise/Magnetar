## Context

The prior three chantiers this session (GPTQ, then AWQ+BitsAndBytes) verified everything against real checkpoint bytes. "Additional Providers" cannot follow that norm literally for Metal/ROCm/NPU/TPU: this repository's tooling has zero hardware for any of them. The user was asked explicitly (`AskUserQuestion`) how to proceed and chose "unverified Metal/ROCm skeletons" -- real code, honestly labeled as untested against real hardware, rather than skipping the item or fabricating verification. Mid-chantier, real technical input relayed from the user's own technical expert reframed the approach twice:

1. First: use `wgpu`/WebGPU instead of hand-written Metal FFI, since `wgpu` compiles to Vulkan on Linux/CI (testable via Lavapipe/software rasterization) and to Metal on macOS automatically -- turning an otherwise-unverifiable Metal path into a genuinely testable one.
2. Second, after the user's own follow-up question surfaced that inference is memory-bound (decode) more than compute-bound (prefill), narrowing where a WGPU/native-Metal gap would actually matter: WGSL/`wgpu` cannot reach Apple Silicon's `simdgroup_matrix` MMA instructions or Metal Performance Shaders' route to the AMX coprocessor -- real for compute-bound prefill, not for memory-bound decode. The user's explicit resulting instruction: keep a Metal Provider crate for that real future need, marked untested, with a call for contributors.

## Goals / Non-Goals

**Goals:**
- Real device discovery for ROCm, genuinely exercised (not mocked) against the real HIP runtime API shape, gracefully reporting unavailable on every host this repository can test.
- An honest Metal placeholder that does not claim untested capability, with clear documentation of what real future work looks like and why it still matters despite `providers/wgpu`.
- A real, hardware-verified WGPU Provider: genuine adapter/device discovery and at least one genuinely dispatched, correctness-verified compute Kernel, on this repository's own real GPU.
- CI wiring that gives the WGPU Provider a real chance to exercise its actual device/compute path in CI (not just its documented fallback), via a real software Vulkan driver.

**Non-Goals:**
- Real Metal, ROCm, NPU, or TPU hardware verification -- none of this hardware exists anywhere in this repository's tooling; this change does not claim otherwise.
- Full `ProviderExecutionApi`/Kernel Registry dispatch integration for any of the three new Providers -- `providers/cpu`/`providers/cuda` parity is out of scope; WGPU's `add` is directly callable, not registry-dispatched.
- The rest of the required Operator set (`matmul`/`rmsnorm`/`rope`/`attention`/`silu`/`residual-add`/...) for WGPU -- one real Kernel establishes the baseline; the rest is real, tracked future work.
- Multi-device execution/placement policy -- this change adds Provider skeletons, not a Multi-Device Placement capability increment.

## Decisions

- **ROCm: dynamic loading via `libloading`, not the `rocmrc` crate.** `rocmrc` exists and is modeled after `cudarc`, but has only 255 downloads and a single maintainer -- weak supply-chain trust for a native-library-adjacent dependency, the exact category this repository has been cautious about elsewhere (`providers/cuda`'s own `deny.toml` precedent). `libloading` is a mature, minimal, widely-used primitive; this crate owns its own tiny HIP FFI surface (`hip_sys.rs`) directly.
- **Metal: no platform `#[cfg]` branch, unconditionally `false`.** A real macOS implementation cannot be written blind and marked "verified" -- an unconditional placeholder, honestly labeled, is more truthful than a `#[cfg(target_os = "macos")]` branch containing untested FFI that looks real but has never compiled or run anywhere.
- **WGPU as the real near-term path to Apple GPUs, Metal kept as real future work, not superseded.** The first round of expert input suggested WGPU could replace the need for a Metal Provider outright; the second round's `simdgroup_matrix`/AMX/MPS finding corrected that -- WGPU is expected to reach near-native performance for memory-bound decode but not compute-bound prefill on Apple Silicon specifically, since WGSL cannot address the matrix-multiply-accumulate hardware path at all. Both crates' READMEs were revised to state this honestly, and the user explicitly instructed keeping `providers/metal` alive rather than deprecating it.
- **WGPU's `add` Kernel is a directly-callable method, not wired into `ProviderExecutionApi`.** Full parity with `providers/cuda` (~7274 lines) was judged out of scope for this chantier; this mirrors the precedent `providers/cuda`'s own half-precision compute set already established (`add-native-cuda-half-precision-compute` landed a real Kernel first; `wire-cuda-half-precision-into-kernel-registry-dispatch` wired it into dispatch as a separate, later chantier).
- **CI installs `mesa-vulkan-drivers` specifically for WGPU.** Without a real Vulkan ICD, GitHub Actions' `ubuntu-latest` runner would only ever exercise WGPU's graceful-unavailability fallback, never its real device-discovery/compute path -- Lavapipe (`llvmpipe`) is a real, if software-only, Vulkan implementation that lets CI genuinely dispatch the compute shader, not just prove the crate compiles.
- **WGPU tests skip gracefully, never hard-assert device availability.** A hard `assert!(is_available())` would fail outright (not skip) on any GPU-less host encountered before Lavapipe was even considered -- fixed via a `require_device_or_skip!` macro mirroring `providers/cuda`'s own benchmark's established graceful-unavailability tolerance, self-caught before this ever reached CI.

## Risks / Trade-offs

- **The `mesa-vulkan-drivers` CI step is unverified as of this change's implementation** -- whether Lavapipe genuinely yields a usable `wgpu` adapter on `ubuntu-latest`, or the tests fall back to graceful skipping, was confirmed only once this change's own CI run completed (see `tasks.md`).
- **WGPU's real Metal-backend behavior remains entirely unverified** -- this change's confidence in Apple GPU support rests on `wgpu`'s own published cross-backend consistency guarantees, not on anything this repository's tooling can independently confirm.
- **ROCm's real HIP API correctness remains entirely unverified against real AMD hardware** -- the FFI signatures are believed correct against HIP's public, CUDA-driver-API-compatible documentation, but no ROCm runtime has ever actually executed this code.
