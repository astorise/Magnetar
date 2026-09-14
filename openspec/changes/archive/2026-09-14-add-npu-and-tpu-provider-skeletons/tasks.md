## 1. Research

- [x] 1.1 Fetched the real, public `ze_api.h` header (`oneapi-src/level-zero`) directly to get the exact `zeInit`/`zeDriverGet` signatures and the real `ZE_INIT_FLAG_VPU_ONLY` flag value, rather than recalling from memory.
- [x] 1.2 Confirmed the real Level Zero loader library names per platform (`ze_loader.dll` on Windows, `libze_loader.so.1` on Linux).
- [x] 1.3 Fetched the real, public `edgetpu_c.h` header (`google-coral/libedgetpu`) directly to get the exact `edgetpu_device_type`/`edgetpu_device`/`edgetpu_list_devices`/`edgetpu_free_devices` declarations.
- [x] 1.4 Confirmed the real Edge TPU runtime library names per platform (`edgetpu.dll` on Windows, `libedgetpu.so.1` on Linux, `libedgetpu.1.dylib` on macOS).

## 2. `providers/npu`

- [x] 2.1 Bootstrapped `Magnetar-provider-NPU` (genuinely empty GitHub repo; cloned into scratchpad, committed, pushed to establish `main` before `git submodule add` could succeed).
- [x] 2.2 `ze_sys.rs`: real Level Zero FFI declarations (`zeInit`, `zeDriverGet`, `ZE_RESULT_SUCCESS`, `ZE_INIT_FLAG_VPU_ONLY`, real per-platform library candidate names).
- [x] 2.3 `provider.rs`: `NpuProvider` dynamically loads the real Level Zero loader via `libloading`, calls the real entry points with `ZE_INIT_FLAG_VPU_ONLY`, reports `Unavailable` on any failure step -- mirrors `RocmProvider`'s graceful-unavailability pattern.
- [x] 2.4 `deny.toml` for `libloading` (mirrors `providers/rocm/deny.toml`); verified via `cargo deny check` (exit 0, advisories/bans/licenses/sources all ok).
- [x] 2.5 `cargo test`/`cargo fmt`/`cargo clippy --all-targets -- -D warnings` all clean (3 tests passed).

## 3. `providers/tpu`

- [x] 3.1 Bootstrapped `Magnetar-provider-TPU` (same empty-repo-then-push sequence).
- [x] 3.2 `edgetpu_sys.rs`: real Edge TPU FFI declarations (`EdgetpuDevice` struct, `edgetpu_list_devices`, `edgetpu_free_devices`, real per-platform library candidate names).
- [x] 3.3 `provider.rs`: `TpuProvider` dynamically loads the real `libedgetpu` runtime via `libloading`, calls `edgetpu_list_devices` and frees the returned array via `edgetpu_free_devices`, reports `Unavailable` on any failure step.
- [x] 3.4 `deny.toml` for `libloading`; verified via `cargo deny check` (exit 0).
- [x] 3.5 `cargo test`/`cargo fmt`/`cargo clippy --all-targets -- -D warnings` all clean (3 tests passed).

## 4. Wiring and CI

- [x] 4.1 `.gitmodules`: two new submodule entries (`providers/npu`, `providers/tpu`).
- [x] 4.2 `.github/workflows/quality.yml`: `provider-integration` job builds/tests both, adds `cargo deny` checks for both; `submodule-integration` job's full sweep includes both `Cargo.toml`s.
- [x] 4.3 `SUBMODULES.md`: Modules table and Compatibility matrix updated for both.
- [x] 4.4 Committed and pushed the main repository's wiring (`.gitmodules`, `quality.yml`, `SUBMODULES.md`, submodule gitlink pins).
- [x] 4.5 Verified real CI green on this commit.

## 5. Documentation

- [x] 5.1 `openspec validate add-npu-and-tpu-provider-skeletons --strict` passes.
- [x] 5.2 README.md's top-level scope-charter status updated once archived: NPU and TPU items closed with an honest status (real device-discovery skeletons, unverified against real hardware).
