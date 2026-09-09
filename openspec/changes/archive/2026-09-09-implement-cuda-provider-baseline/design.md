## Context

`providers/cuda` is an independent git submodule repository
(`Magnetar-provider-CUDA`), currently the empty `cargo new --lib` template
(`SUBMODULES.md`). It is built and tested by CI's `submodule-integration` job
(`.github/workflows/quality.yml`) via `cargo test --locked --manifest-path
providers/cuda/Cargo.toml` on plain `ubuntu-latest` runners — **no GPU, no
NVIDIA driver, no CUDA Toolkit**. That constraint is the single biggest
influence on this design: whatever `providers/cuda` becomes, it must still
build and pass `cargo test` on that runner, or CI turns red for everyone,
regardless of this repository's own working machine (RTX 3070, CUDA Toolkit
13.2, confirmed present locally).

`providers/cpu` is the direct architectural precedent: a plain Rust crate with
a path dependency on `magnetar-runtime`, implementing the `Provider` trait
in-process (`ReferenceCpuProvider`/`ReferenceCpuExecutor` in
`providers/cpu/src/lib.rs`), free-function kernels (`matmul`, `rmsnorm`,
`rope`, `attention`, `softmax_rows`, `silu`, `add`, `mul`, ...), and metadata
builders (`reference_cpu_provider_metadata`, `reference_cpu_device`,
`reference_cpu_kernel_advertisements`). `magnetar-runtime::provider` already
defines the full contract this must plug into: `Provider`,
`ProviderExecutionApi`, `ProviderMetadata`, `ProviderError`, plus the dynamic
ABI types (`ProviderAbiDescriptor` etc.) that this baseline deliberately does
not use yet.

Unlike `reference-cpu` (a correctness-first *baseline*), `provider-roadmap`
frames CUDA as an *optimized* Provider phase: it must preserve
`reference-cpu`'s portable Operator semantics and pass conformance against it
within tolerance, but it is allowed — expected — to actually be fast, and it
is explicitly not the correctness oracle.

Since this design was first drafted, a self-hosted GPU-equipped CI lane has
been added: `arc-gpu-magnetar`, an Actions Runner Controller scale set on the
Talos cluster, exercised by `.github/workflows/gpu-runner-smoke.yml`
(`workflow_dispatch`-only, not yet run per-PR). Its job runs inside a
`nvidia/cuda:13.x-cudnn-devel-ubuntu24.04` container with `--gpus all` (image
tag kept current by Renovate — see `renovate/nvidia-cuda-13.x`).

This lane shares GPU capacity with Tachyon workloads on the same cluster, and
that contention is real: the first three dispatches (runs `33956362434`,
`33958109055`, `33960577838`) each failed after the ARC hook's ~10-minute
wait with `pod arc-gpu-magnetar-...-workflow is unhealthy with phase status
Pending` — the job container never got scheduled, so no step past
"Initialize containers" ran. Once infra freed the GPU, a subsequent dispatch
(run `33961785474`) **succeeded end-to-end**: `nvidia-smi` reported an
NVIDIA GeForce RTX 3060 (driver 595.71.05, CUDA 13.3), `nvcc --version`
resolved `/usr/local/cuda/bin/nvcc`, `CUDA_HOME` existed, and `cargo test
--locked --manifest-path providers/cuda/Cargo.toml` passed against today's
empty template. So the lane itself, the submodule checkout, and the pinned
Rust toolchain install all work — the open risk is purely that dispatches can
queue or fail with `Pending` while Tachyon holds the GPU, not anything about
the runner's own configuration. A `Pending`-timeout dispatch should be
retried once GPU capacity is confirmed free, not read as a signal about
`providers/cuda` itself.

This means real hardware conformance for this change has two available
venues rather than one: this workstation (RTX 3070, CUDA Toolkit 13.2,
confirmed reachable locally) for fast local iteration, and `arc-gpu-magnetar`
for an in-CI check on different hardware (RTX 3060) — contention permitting.
Given the shared, contended nature of the cluster GPU, this baseline still
does not depend on `arc-gpu-magnetar` being available on demand (see
Migration Plan and Open Questions), but tasks.md now includes running the
real conformance suite there as well once implemented, not only locally.

## Goals / Non-Goals

**Goals:**
- A `CudaProvider` that registers as a built-in Provider, discovers real CUDA
  Devices when a compatible driver is present, and executes the
  `operator-scope` required-now kernels on GPU for f32/contiguous tensors.
- `providers/cuda` keeps building and passing `cargo test --locked` on a
  GPU-less, CUDA-Toolkit-less `ubuntu-latest` runner, unmodified from today's
  CI job definition.
- On a machine that does have a compatible GPU and driver (this workstation),
  the same binary actually executes kernels and can be conformance-tested
  against `reference-cpu`.
- No native CUDA handle (context, stream, device pointer, module) ever
  crosses into `magnetar-runtime`'s public Device/Tensor metadata.

**Non-Goals:**
- Dynamic-library Provider ABI loading — `CudaProvider` is built-in (Rust
  trait object), same loading mode `providers/cpu` uses today.
- The Execution Stream ABI extension — kernel calls are synchronous from the
  caller's perspective for this baseline (see Decisions, "no async streams
  yet").
- The Device Memory Pool contract's soft/hard reservation and watermark
  machinery — this baseline does direct allocate/free per buffer.
- Multi-GPU placement, quantization, flash/paged attention, non-f32 dtypes,
  non-contiguous layouts.
- A GPU-equipped CI runner. Real-hardware conformance stays a local/manual
  activity for this change, the same way `implement-model-format-parsers`
  left real `cargo-fuzz` execution as local/periodic rather than per-PR CI.

## Decisions

### Decision: `cudarc` in dynamic-loading mode, not link-time CUDA, not `cust`/`rustc_codegen_nvvm`

Use the `cudarc` crate (safe Rust wrapper over the CUDA driver API, NVRTC, and
optionally cuBLAS) with its dynamic-loading feature: it `dlopen`s
`libcuda.so`/`nvcuda.dll` and `libnvrtc.so`/`nvrtc64_*.dll` at *runtime*
instead of linking against CUDA import libraries at *build* time.

Alternatives considered:
- **`cust` / `rustc_codegen_nvvm` (Rust-CUDA project)**: writes kernels in
  Rust, compiled through a custom NVVM codegen backend requiring a specific
  nightly toolchain and LLVM/NVVM setup. Far heavier toolchain risk on
  Windows, and still needs the CUDA Toolkit at build time — incompatible with
  the GPU-less CI runner.
- **Link-time `cudarc` (default feature) or raw `-lcuda` FFI**: fails to
  *link* on the CI runner because `libcuda`/`nvcuda` isn't installed there at
  all (it ships with the driver, not the toolkit) — this breaks `cargo build`
  itself, not just runtime behavior. Unacceptable given the CI constraint.
- **`build.rs` + `nvcc`, precompiled PTX/cubin checked in or built ahead of
  time**: avoids the link problem but still needs `nvcc` at build time unless
  PTX is checked into the repo as a binary artifact (undesirable — it
  couples source control to a specific CUDA Toolkit version and architecture
  target, and `nvcc`-less contributors couldn't rebuild it). Rejected in
  favor of runtime NVRTC compilation (next decision).

Why this works for CI: `cudarc`'s dynamic-loading mode has zero link-time or
build-time dependency on any CUDA component. `cargo build`/`cargo test`
succeed on the GPU-less runner; only the runtime *attempt* to open the driver
library fails there, which this design turns into a normal "Provider
unavailable" outcome (see next decision) instead of a build or test failure.

### Decision: kernels are CUDA C++ source compiled to PTX via NVRTC at first Provider use, not ahead-of-time

Kernel bodies (`matmul`, `embedding_lookup`, `rmsnorm`, `rope`, `attention`,
`softmax`, `silu`, `add`/`mul`/`residual_add`) live as `.cu` string constants
in `providers/cuda/src/kernels/`, compiled to PTX through `cudarc`'s NVRTC
binding the first time `CudaProvider` initializes on a machine that has a
usable driver, then loaded as a CUDA module and cached for the process
lifetime.

This is the mechanism that keeps kernel source human-readable and
conformance-testable (a `.cu` file, not an opaque prebuilt blob) while
requiring nothing at build time. It is explicitly *not* the versioned,
cross-Provider Kernel Compilation ABI extension from `provider-abi` — that
extension is about a Runtime-facing contract for Providers to expose
compilation as a service; this is `CudaProvider`'s own private
implementation detail, same as `providers/cpu` privately choosing how it
loops over floats.

### Decision: graceful unavailability when no compatible driver is present

`CudaProvider::new()` (or equivalent constructor) attempts to load the CUDA
driver via `cudarc` and enumerate devices. If that fails for any reason (no
driver, no compatible GPU, driver/runtime version mismatch), `CudaProvider`
still constructs successfully but:
- reports zero Devices,
- reports Provider Health as `unavailable` (not `failed` — this is expected,
  policy-relevant absence of hardware, not an internal fault), consistent
  with `provider`'s Health Model distinguishing `unavailable` from `failed`,
- is skipped by Resolution the same way `provider`'s "Provider Fallback" and
  "Provider Isolation" requirements already describe for any Provider that
  cannot execute a requested Capability.

This is what makes the GPU-less CI runner pass: `cargo test` exercises this
exact path (driver load fails in CI, `CudaProvider` reports unavailable, unit
tests assert that reporting is correct) instead of skipping CUDA tests via
`#[ignore]` or `cfg` tricks. The *only* thing gated behind actual hardware is
executing a real kernel and comparing its output to `reference-cpu` — those
tests are marked to skip (not fail) when device enumeration returns zero
Devices, and this workstation is expected to be where they actually run.

### Decision: no async Execution Stream extension yet — synchronous per-call execution

Each `ProviderExecutionApi` call this baseline handles launches its kernel(s)
on a single CUDA stream and synchronizes (blocks) before returning, rather
than returning a pending completion token per `execution-stream`. `provider-
abi`'s "Synchronization ABI Extension Is Optional" / "Existing v1 Provider"
requirements explicitly allow this: a Provider with only synchronous
execution remains valid. Overlap between host and device work, and
cross-stream ordering, are deferred to a follow-up CUDA phase once the
baseline's correctness is established — mirroring how `reference-cpu`
shipped before `execution-stream` existed at all.

### Decision: direct `cuMemAlloc`/`cuMemFree` per buffer, not the Device Memory Pool contract

`device-memory-pool` defines soft/hard reservations and watermark-driven
pressure; implementing that fully is significant scope on its own (it already
has its own OpenSpec history as a dedicated change). This baseline instead
allocates device memory directly per Tensor Resource and frees it on release,
reporting a simple free-memory-derived pressure estimate through Device
metadata (`Device SHALL Expose Pressure Estimate`) without implementing
pooling. This satisfies `device`'s descriptive requirements without claiming
the full pool contract. Follow-up work can introduce pooling without changing
`CudaProvider`'s external Device/Provider contract.

### Decision: conformance scope for this change is `provider-core` + `provider-compute` only

`provider`'s Conformance Profiles list six profiles
(`provider-core`, `provider-compute`, `provider-data-movement`,
`provider-cancellation`, `provider-observability`, `provider-dynamic-abi`).
This baseline targets the two that are meaningful without the Execution
Stream extension and dynamic ABI: `provider-core` (metadata, identity,
Capability advertisement, Device metadata, lifecycle/status, error mapping)
and `provider-compute` (the actual kernels, numerically checked against
`reference-cpu`). `provider-dynamic-abi` is inapplicable (built-in Provider);
`provider-cancellation` and richer `provider-data-movement` scenarios are
deferred with the async-stream follow-up; `provider-observability` baseline
redaction is covered incidentally by reusing `magnetar-runtime`'s existing
observability plumbing, not a CUDA-specific mechanism.

## Risks / Trade-offs

- **[Risk]** NVRTC-compiled kernels are unfused, non-tuned CUDA C++ — likely
  far from cuBLAS/cuDNN performance. → Mitigation: this baseline's purpose is
  a correct, conformant first cut, not a fast one; `provider-roadmap`
  explicitly separates benchmarks from conformance ("Benchmarks Do Not
  Replace Conformance"), so this is an accepted trade-off, not a defect.
  Fused/tuned kernels or cuBLAS-backed matmul are natural follow-up phases.
- **[Risk]** `cudarc`'s dynamic-loading mode is a third-party dependency whose
  maintenance and CUDA-version-compatibility surface is out of this repo's
  control. → Mitigation: it is MIT/Apache-2.0 (compatible with `deny.toml`'s
  allowed-license list) and isolated entirely inside `providers/cuda`;
  `magnetar-runtime` never depends on it, so a future replacement only
  touches this one submodule.
- **[Risk]** The default `submodule-integration` CI job (plain `ubuntu-latest`)
  never exercises a real kernel launch, so a regression that only manifests
  on real hardware (e.g., a genuine NVRTC compile error, a wrong kernel
  launch configuration) can merge unnoticed through that job alone.
  → Mitigation: `arc-gpu-magnetar` is now confirmed reachable and working
  (see Context, run `33961785474`) as an additional, on-demand real-hardware
  lane, and tasks.md has both this workstation and `arc-gpu-magnetar` as
  required verification steps before this change is considered done. It
  remains a *supplement*, not a per-PR gate, because it shares contended GPU
  capacity with Tachyon and a `Pending`-timeout dispatch is expected
  occasionally rather than treated as a CUDA Provider failure.
- **[Trade-off]** Skipping the Device Memory Pool and Execution Stream
  contracts means `CudaProvider` cannot yet overlap transfers with compute or
  bound its memory footprint under pressure — acceptable for a first
  correctness-focused cut, consistent with `reference-cpu` having shipped
  without those either.

## Migration Plan

1. Implement `CudaProvider` and kernels in `providers/cuda` behind the
   existing submodule boundary; no `magnetar-runtime` or `magnetar-cli`
   changes are required to land this change (built-in Providers still need a
   call site to actually be registered into a running Runtime, but wiring
   `CudaProvider` into `magnetar-cli`'s default Provider set is a separate,
   later change — this change proves the Provider itself, not end-to-end CLI
   selection).
2. Land with `submodule-integration` CI green on the existing GPU-less
   runner (graceful-unavailability path).
3. Manually verify on this workstation (RTX 3070, driver + CUDA Toolkit
   present) that Device discovery finds the GPU and each required-now kernel
   matches `reference-cpu` output within tolerance; capture that as the
   conformance report `provider`'s "Conformance Report" requirement expects.
4. Additionally dispatch `gpu-runner-smoke.yml` (or its successor once it
   runs the real conformance suite instead of the template) against
   `arc-gpu-magnetar` and capture that result too — a second, different-GPU
   (RTX 3060) data point confirmed reachable per Context. Treat a
   `Pending`-timeout dispatch there as inconclusive/retry-later, not as a
   CUDA Provider failure, given the cluster's shared GPU capacity with
   Tachyon.
5. Update `SUBMODULES.md`'s `providers/cuda` row and compatibility matrix
   once merged, following `providers/cpu`'s existing entry format.
6. Rollback: revert the submodule pin bump in this repository (per
   `SUBMODULES.md`, this repo pins one exact commit) — no schema/data
   migration involved since nothing is persisted across this change.

## Open Questions

- Should `providers/cuda`'s device-pressure estimate use `cuMemGetInfo`
  free/total bytes directly, or should it round/bucket the value to avoid
  encouraging callers to treat it as exact accounting? (`device`'s "Device
  SHALL Expose Pressure Estimate" only requires an estimate.)
- `arc-gpu-magnetar` is now confirmed working (run `33961785474`, after three
  prior `Pending`-timeout dispatches while Tachyon held the GPU — see
  Context), so it is a real, usable, but *contended* lane rather than a
  guaranteed-available one. Once this change lands real content, should
  `provider-compute` conformance run there per-PR (path-filtered on
  `providers/cuda/**`), stay manual dispatch, or run on a schedule, given it
  competes with Tachyon for the same GPU capacity? Out of scope to decide
  here, but worth resolving before treating it as a required gate anywhere.
