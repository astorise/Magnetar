## Why

`providers/cuda` (`Magnetar-provider-CUDA`) is still an empty `cargo new --lib`
template (`SUBMODULES.md`). `provider-roadmap` already commits Magnetar to a
CUDA Provider phase with conformance gates, and `reference-cpu` already exists
as the correctness oracle CUDA output must match within tolerance, but no
Provider actually executes on a GPU yet. Local inference on this workstation
(RTX 3070, CUDA Toolkit 13.2 present) is CPU-only until a first real CUDA
Provider exists. This change gives Magnetar its first hardware-accelerated
execution path and the concrete Provider that the roadmap's CUDA phase
scenarios have been describing in the abstract.

## What Changes

- Implement a real `CudaProvider` in `providers/cuda` (replacing the empty
  template) that registers with `magnetar-runtime`'s Provider Registry as a
  built-in Provider (Rust trait object, no dynamic-library ABI yet) and
  advertises `magnetar:compute/run`.
- Discover and expose at least one CUDA Device through Runtime-owned Device
  metadata (name, compute capability, total/estimated-free memory, pressure
  estimate) without ever exposing a raw CUDA context, stream, or pointer.
- Implement GPU kernels for exactly the `operator-scope` required-now tier
  needed for the first decoder path: embedding lookup, RMSNorm, matmul
  (including logits projection), RoPE (baseline mode only), causal attention,
  softmax, SiLU, add, mul, residual-add — f32 only, contiguous layout only,
  matching `reference-cpu`'s portable semantics within declared tolerance.
- Add explicit host<->device data movement (upload/download) with no silent
  dtype/layout conversion and no silent CPU fallback; unsupported
  dtype/layout/shape requests fail with structured errors.
- Wire CUDA-allocated device memory into Runtime Memory Manager accounting and
  Resource Affinity (Device-resident, Provider-pinned) using a straightforward
  allocate/free strategy — the full Device Memory Pool (soft/hard reservations,
  watermarks) and Device-Resident zero-copy/replica contracts are out of scope
  for this baseline and are left for a follow-up change.
- Add a `provider-core` + `provider-compute` conformance run for `CudaProvider`
  against the existing Provider Conformance Suite, using `reference-cpu` as the
  numerical oracle.
- Update `SUBMODULES.md`'s `providers/cuda` row from "Empty `cargo new --lib`
  template" to a real-content description, and pin the module's compatibility
  entry once implemented.

**Explicit non-goals for this baseline** (left for later CUDA-phase changes
per `provider-roadmap`):
- Dynamic-library Provider ABI loading (`provider-abi`, `Provider ABI
  Descriptor`) — `CudaProvider` ships built-in for now.
- The asynchronous Execution Stream ABI extension (`execution-stream`) —
  kernels run synchronously per call for this baseline.
- The Kernel Compilation ABI extension (`provider-abi`) — this baseline's own
  internal NVRTC compilation (see design.md) is not exposed as the versioned
  cross-Provider Compilation ABI extension.
- Multi-GPU / `multi-device-placement` — only single-Device discovery and
  dispatch.
- Quantized execution, flash/paged attention, non-f32 dtypes, non-contiguous
  layouts — all remain explicitly unsupported and fail with structured errors.

## Capabilities

### New Capabilities

- `cuda-provider`: the CUDA Provider baseline analogous to `cpu-provider` —
  registration/identity, Device discovery, required-now GPU kernel coverage,
  explicit host/device data movement, dtype/layout/error declarations,
  Memory Manager integration, and its role relative to `reference-cpu` as
  correctness oracle.

### Modified Capabilities

- None. `provider`, `provider-roadmap`, `provider-abi`, `device`,
  `operator-scope`, and `reference-cpu` already describe the contract a CUDA
  Provider must satisfy; this change delivers an implementation that conforms
  to those existing requirements rather than changing them. If conformance
  testing surfaces a genuine gap in those specs, it will be handled as a
  separate spec-change, not folded into this one.

## Impact

- `providers/cuda/Cargo.toml`, `providers/cuda/src/**`: real implementation
  replacing the template; adds a path dependency on `magnetar-runtime` (same
  pattern as `providers/cpu`) plus a CUDA host-binding crate used in
  dynamic-loading mode (no link-time dependency on the CUDA driver or CUDA
  Toolkit — see design.md).
- `magnetar-runtime`: no contract changes expected; `CudaProvider` consumes
  existing `Provider`/`ProviderExecutionApi`/Memory Manager/Device APIs. Any
  gap found is a separate change.
- No new build-time dependency: `providers/cuda` continues to build with only
  the pinned Rust toolchain, matching the existing `submodule-integration` CI
  job which runs `cargo test --locked --manifest-path providers/cuda/Cargo.toml`
  on plain `ubuntu-latest` (no GPU, no CUDA Toolkit installed). GPU-dependent
  behavior degrades to a reported-unavailable Provider on that runner instead
  of failing the build (see design.md).
- `SUBMODULES.md`: `providers/cuda` row and compatibility matrix updated.
- CI: default `submodule-integration` coverage stays GPU-less (asserts
  graceful unavailability); real-GPU conformance runs only where hardware is
  present (addressed in design.md as a follow-up, not blocking this change).
