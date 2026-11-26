## Why

This is lot 1 of the [accelerator packages RFC](../../../docs/rfcs/accelerator-packages-2026-11-23.md). Its goal is the first hardware-verified execution of a kernel Magnetar did not build into a Provider, through the Runtime's normal path:

> bundle → ingestion → trust and permission → qualification on a real GPU → Provider preparation → registry candidate → selection → dispatch

Today that path stops before the Provider:

- The CUDA Provider compiles its embedded CUDA source with NVRTC at initialization and loads the resulting PTX (`providers/cuda/src/kernels.rs`).
- Its kernels are advertised statically (`cuda_kernel_advertisements`), and invocations are routed by kernel name (`executor.rs`).
- `ProviderKernelCompilationApi` (source → compiled artifact) exists as a contract, but no Provider implements it.
- Nothing lets a Provider take an already compiled artifact, load it, and expose it as a kernel the registry can select.

Loading a precompiled CUBIN is not compiling a source, and the RFC asks to keep the two apart. A CUBIN also needs a launch contract: entry point, parameter layout and grid. The manifest does not carry one today.

## What Changes

- **Provider Kernel Preparation**, an optional capability (`magnetar-provider-api` and `magnetar-kernel`), separate from compilation:
  - `kernel_preparation_api()` returns a `ProviderKernelPreparationApi` that advertises the formats and targets it accepts;
  - `prepare` takes an accepted, trusted, permitted and qualified compiled artifact (Runtime-staged bytes), its launch descriptor and a target Device, and returns an opaque prepared kernel identity and a kernel advertisement;
  - `release` frees it;
  - preparation is cold path only and never compiles source.
- **CUDA launch descriptor**: a required manifest extension `magnetar:cuda-launch@1`. It declares:
  - the entry symbol;
  - the ordered parameters, each one of: an input or output tensor (a device pointer), an integer or float attribute, or a derived value (a dimension or an element count);
  - a fixed block size, a grid derived by ceil-division of a derived value, and dynamic shared memory.

  These are declarative values in a closed vocabulary, never code.
- **CUDA Provider preparation** of `nvidia:cubin` images whose architecture matches the Device's compute capability:
  - the image is loaded from memory through the driver, never from a path the manifest names;
  - the entry symbol is resolved and the descriptor checked against it;
  - the prepared kernel is launched on the Provider's stream with device-resident tensors.

  PTX (JIT by the driver) is deferred.
- **Registry candidates from prepared artifacts.** After preparation, the Provider registers the advertisement under Provider authority, with its artifact binding (bundle identity, blob digest, generation). The built-in kernel stays registered as the known-good fallback. Selection uses normal policy; the experiment uses an explicit, auditable pin.
- **Step captures and modules.** A CUDA graph capture that recorded a launch of a prepared kernel keeps its module loaded: `release` reports the kernel in use while a capture references it. Lot 2 generalizes this into generation draining.
- **The experiment:**
  - `silu-mul`, then `rmsnorm`, as checked-in CUDA sources compiled to an `sm_86` CUBIN with NVRTC on the GPU runner;
  - packaged as a Kernel Exchange Bundle with its launch descriptor and signed with a CI key that only the test trust store trusts;
  - then ingested, qualified against the Reference CPU kernel, prepared, pinned and run in a real-checkpoint decode.

  TTFT, TPOT, p50/p95, memory and preparation time are compared with the built-in kernel.

## Capabilities

### New Capabilities

- `provider-kernel-preparation`: the optional Provider capability that turns an admitted compiled artifact into a prepared, dispatchable kernel.
- `cuda-kernel-launch-descriptor`: the `magnetar:cuda-launch@1` manifest extension.

### Modified Capabilities

- `cuda-provider`: preparation of `nvidia:cubin` images (added requirements).
- `kernel-registry`: candidates backed by Provider-prepared artifacts (added requirements).

## Impact

- `magnetar-provider-api`: an optional `kernel_preparation_api()` on `Provider` (default `None`, so every Provider stays valid).
- `magnetar-kernel`: preparation request and result types; the launch descriptor extension's parsing and validation.
- `magnetar-runtime`: the preparation step after eligibility (lot 0), and candidate registration.
- `providers/cuda`: in-memory module loading, symbol resolution, generic launch from a descriptor, capture references.
- `tools/kernel-bundle-fixture` (new, test tooling): NVRTC to CUBIN, bundle packaging and signing for CI.
- `.github/workflows/gpu-runner-smoke.yml`: the experiment's job.
- No WIT change, no Component change.

## Dependencies

- Lot 0: [`define-kernel-artifact-sources`](../define-kernel-artifact-sources/proposal.md) (sources, trust store, execution permission, lifecycle).
- C09 pinned selection in [`select-kernels-from-measured-context`](../select-kernels-from-measured-context/proposal.md): an explicit pin suffices until C09 lands.

## Compatibility and authority

- The CUDA Provider's built-in kernels, advertisements and step capture are unchanged when no external artifact is prepared.
- Preparation only happens after the Runtime decides eligibility. The Provider receives bytes and a descriptor, never a path or URL.
- A Provider that does not implement preparation keeps returning `None`.

## Non-goals

- Loading PTX or compiling source in this path (`ProviderKernelCompilationApi` stays the compilation contract).
- Unloading a module in use, hot swap or rollback beyond the registry's existing generation calls (lot 2).
- Kernels whose launch geometry cannot be expressed declaratively (a kernel needing host-side planning stays built in).
- Any claim of speedup before the measurement.

## Companion changes

The delta to a capability that now lives in its own repository moved there, with a companion change of the same name; this change keeps the rest.

- `cuda-provider`: the delta is in https://github.com/astorise/Magnetar-provider-CUDA/tree/HEAD/openspec/changes/prepare-prebuilt-cuda-kernels/ (companion change of the same name)
