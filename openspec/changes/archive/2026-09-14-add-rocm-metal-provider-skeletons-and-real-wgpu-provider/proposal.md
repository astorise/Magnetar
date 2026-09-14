## Why

The original scope charter's remaining large item after quantization (GPTQ/AWQ/BitsAndBytes, already closed) was multi-device execution and additional Providers (Metal/ROCm/NPU/TPU). This repository's own tooling has exactly one real GPU (an NVIDIA laptop GPU) and no Apple Silicon, AMD, NPU, or TPU hardware anywhere -- so "implement and verify on real hardware" the way `providers/cpu`/`providers/cuda` were is not possible for Metal or ROCm specifically. Rather than skip the item or fake verification, this change draws an explicit, honest line: real device-discovery skeletons for ROCm and Metal that gracefully report unavailable (never claiming compute correctness never tested), plus a real, hardware-verified GPU compute Provider built on `wgpu` -- which, unlike hand-written Metal FFI, is independently testable on this repository's real Vulkan-capable hardware and (per real technical input relayed mid-session) is expected to carry that verification over to Apple's Metal backend for memory-bound workloads, with an honestly documented gap for compute-bound workloads that still need native Metal (`simdgroup_matrix`/AMX/MPS access `wgpu` cannot reach).

## What Changes

- New submodule `providers/rocm` (`Magnetar-provider-ROCm`): `RocmProvider` dynamically loads the real HIP runtime library (`amdhip64.dll`/`libamdhip64.so{,.6,.5}`) and calls the real `hipInit`/`hipGetDeviceCount` entry points, gracefully reporting `ProviderHealth::Unavailable` when no ROCm runtime is found -- true everywhere in this repository's tooling. No compute Kernels (no AMD hardware to implement or verify them against).
- New submodule `providers/metal` (`Magnetar-provider-Metal`): `MetalProvider` is an honest, unconditionally-unavailable placeholder -- no `#[cfg(target_os = "macos")]` branch at all, since no macOS environment exists anywhere in this repository's tooling to implement or verify real Metal FFI against. README documents `providers/wgpu` as the real near-term path to Apple GPUs and this crate as real future work specifically for the compute-bound kernels `wgpu` cannot reach, with a call for contributors on real Apple Silicon hardware.
- New submodule `providers/wgpu` (`Magnetar-provider-WGPU`): `WgpuProvider` does real, hardware-verified adapter/device discovery and one real compute Kernel (`add`) via `wgpu` (Vulkan/Metal/DX12) -- genuinely exercised on this repository's real NVIDIA GPU through the real Vulkan backend, conformance-tested against `providers/cpu::add`. Directly callable (`WgpuProvider::add`), not yet wired into `ProviderExecutionApi`/Kernel Registry dispatch (deliberately deferred, matching `providers/cuda`'s own half-precision "Kernel first, dispatch wiring later" sequencing).
- `.github/workflows/quality.yml`: `provider-integration` job builds/tests all three new Providers, installs `mesa-vulkan-drivers` (a real software Vulkan ICD, `llvmpipe`) so the GPU-less CI runner can exercise WGPU's real device-discovery/compute path rather than only its fallback, and runs `cargo deny` for ROCm's and WGPU's new dependencies (`libloading`, `wgpu`/`pollster`); `submodule-integration` job's full test sweep includes all three.
- **BREAKING**: none. All three are new, independent submodules; no existing crate's public contract changes.

## Capabilities

### New Capabilities
- `rocm-provider`: real device-discovery-only Provider baseline, unavailable everywhere this repository's tooling can verify.
- `metal-provider`: unconditionally-unavailable Provider placeholder, real future work documented.
- `wgpu-provider`: real, hardware-verified cross-platform GPU compute Provider baseline (one Kernel, `add`), not yet the full `ProviderExecutionApi` surface.

### Modified Capabilities
(none)

## Impact

- New repositories: `Magnetar-provider-ROCm`, `Magnetar-provider-Metal` (bootstrapped from a pre-existing empty template), `Magnetar-provider-WGPU`, each pinned into this repository as a git submodule (`providers/rocm`, `providers/metal`, `providers/wgpu`) via `.gitmodules`.
- `.github/workflows/quality.yml`: `provider-integration` and `submodule-integration` jobs extended.
- `SUBMODULES.md`: Modules table and Compatibility matrix updated for all three.
