## 1. Scoping

- [x] 1.1 Asked the user how to proceed given zero Metal/ROCm/NPU/TPU hardware in this repository's tooling (`AskUserQuestion`); user chose unverified Metal/ROCm skeletons over skipping the item.
- [x] 1.2 Asked whether to create new GitHub repositories for the new Providers; user chose yes, both.
- [x] 1.3 Incorporated real technical input relayed by the user (twice) into the approach: pivot to `wgpu` for a genuinely testable cross-platform GPU path, then correct that approach once `simdgroup_matrix`/AMX/MPS limitations were identified, per the user's explicit instruction to keep a Metal Provider crate alive as real future work.

## 2. `providers/rocm`

- [x] 2.1 Bootstrapped `Magnetar-provider-ROCm` (genuinely empty GitHub repo; cloned into scratchpad, committed, pushed to establish `main` before `git submodule add` could succeed).
- [x] 2.2 `hip_sys.rs`: real HIP FFI declarations (`hipInit`, `hipGetDeviceCount`, real per-platform library candidate names).
- [x] 2.3 `provider.rs`: `RocmProvider` dynamically loads the real HIP library via `libloading`, calls the real entry points, reports `Unavailable` on any failure step -- mirrors `CudaProvider`'s graceful-unavailability pattern.
- [x] 2.4 `deny.toml` for `libloading` (mirrors `providers/cuda/deny.toml`); verified via `cargo deny check` (exit 0, advisories/bans/licenses/sources all ok).
- [x] 2.5 `cargo test`/`cargo fmt`/`cargo clippy --all-targets -- -D warnings` all clean (3 tests passed).

## 3. `providers/metal`

- [x] 3.1 Discovered `Magnetar-provider-Metal` already existed as a pre-seeded empty `cargo new --lib` template (`6e43258`); reused it rather than creating a duplicate.
- [x] 3.2 `provider.rs`: `MetalProvider::is_available()` unconditionally `false`, no platform `#[cfg]` branch.
- [x] 3.3 README: Purpose/Status, then revised twice for the two rounds of real technical input -- final state documents `providers/wgpu` as the real near-term Apple GPU path and this crate as real future work for the `simdgroup_matrix`/AMX/MPS-reliant compute-bound kernels `wgpu` cannot reach, with an explicit call for contributors on real Apple Silicon hardware.
- [x] 3.4 `cargo test`/`cargo fmt`/`cargo clippy --all-targets -- -D warnings` all clean (2 tests passed). No `deny.toml` (zero new dependencies, matching `providers/cpu`'s own precedent).

## 4. `providers/wgpu`

- [x] 4.1 Bootstrapped `Magnetar-provider-WGPU` (fresh empty repo, same establish-`main`-then-`submodule add` sequence as ROCm).
- [x] 4.2 `add.wgsl`: real WGSL compute shader (bounds-checked, `workgroup_size(64)`).
- [x] 4.3 `kernels.rs`: real `wgpu` buffer/pipeline/bind-group/dispatch/readback implementation of `add`, iterated against real compiler feedback from `wgpu` 30.0.1's actual API surface (six real compile-error fixes, found by reading the vendored dependency source directly rather than guessing) and two real clippy findings.
- [x] 4.4 `provider.rs`: `WgpuProvider` requests a real adapter/device (headless, no display handle), reports `Unavailable` gracefully on failure, exposes a directly-callable `add` method.
- [x] 4.5 `deny.toml` for `wgpu`/`pollster`; verified via `cargo deny check` (exit 0).
- [x] 4.6 Fixed a self-caught design bug before it reached CI: hardware-dependent tests originally hard-`assert!`ed device availability (would fail, not skip, on a GPU-less host); replaced with a `require_device_or_skip!` macro.
- [x] 4.7 `cargo test`/`cargo fmt`/`cargo clippy --all-targets -- -D warnings` all clean; 5 tests passed on this repository's real NVIDIA GPU (device discovery, correctness against hand-computed values, a non-multiple-of-workgroup-size case, and conformance against `providers/cpu::add`), confirmed stable across repeated runs.
- [x] 4.8 README: documents what is real/verified here (Vulkan backend, on this real GPU) vs. what rests on `wgpu`'s own cross-backend guarantees (Metal on macOS, never run here), plus the `simdgroup_matrix`/AMX/MPS decode-vs-prefill limitation and pointer back to `providers/metal`.

## 5. Wiring and CI

- [x] 5.1 `.gitmodules`: three new submodule entries (`providers/rocm`, `providers/metal`, `providers/wgpu`).
- [x] 5.2 `.github/workflows/quality.yml`: `provider-integration` job builds/tests all three, installs `mesa-vulkan-drivers` before the WGPU step, adds `cargo deny` checks for ROCm and WGPU; `submodule-integration` job's full sweep includes all three `Cargo.toml`s.
- [x] 5.3 `SUBMODULES.md`: Modules table and Compatibility matrix updated for all three.
- [x] 5.4 Committed and pushed the main repository's wiring (`.gitmodules`, `quality.yml`, `SUBMODULES.md`, submodule gitlink pins).
- [x] 5.5 Verified real CI green on this commit, including whether `mesa-vulkan-drivers` genuinely lets the WGPU Provider's tests exercise real hardware on GitHub Actions' `ubuntu-latest` runner rather than gracefully skip -- see `Quality` run 34856742040.

## 6. Documentation

- [x] 6.1 `openspec validate add-rocm-metal-provider-skeletons-and-real-wgpu-provider --strict` passes.
- [x] 6.2 README.md's top-level scope-charter status updated once archived: additional-Providers item closed with an honest status per Provider (ROCm: real discovery skeleton; Metal: honest placeholder with call for contributors; WGPU: real discovery + one real hardware-verified Kernel).
