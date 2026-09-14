## Context

`add-rocm-metal-provider-skeletons-and-real-wgpu-provider` established the pattern this change reuses directly: dynamically load a real, publicly-documented vendor runtime library via `libloading`, call its real, minimal driver-discovery entry points, and gracefully report unavailable on any failure. The open question for NPU/TPU was which real API to target -- both terms cover many possible vendor-specific stacks, and picking the wrong one (or inventing signatures rather than fetching real ones) would repeat the exact "unverifiable guesswork" failure mode this repository's development practice avoids everywhere else.

## Goals / Non-Goals

**Goals:**
- Real device discovery for a genuine, vendor-neutral-where-possible NPU API (Level Zero, also Intel's own GPU discovery API, so this crate's approach generalizes rather than being a one-off).
- Real device discovery for the real, purchasable local Coral Edge TPU, explicitly distinguished from Google Cloud TPU (a cloud service, out of scope for local device discovery by construction).
- Every FFI declaration transcribed from a real, fetched header (`ze_api.h`, `edgetpu_c.h`), not written from memory or inferred from documentation summaries.

**Non-Goals:**
- Real Intel NPU or Coral Edge TPU hardware verification -- neither exists anywhere in this repository's tooling; this change does not claim otherwise.
- Full `ProviderExecutionApi`/Kernel Registry dispatch integration -- device discovery only, matching `providers/rocm`'s own scope boundary.
- Per-device enumeration beyond a raw count (`zeDeviceGet`, or per-device `edgetpu_device` fields) -- real future work for a contributor with real hardware to build and verify against.
- A compute execution model for Edge TPU specifically -- Edge TPU's ahead-of-time-compiled execution model (via a separate Edge TPU Compiler toolchain) does not map onto Level Zero/HIP/CUDA's runtime kernel-source compilation shape, and designing that mapping speculatively, without real hardware to verify against, is explicitly deferred.

## Decisions

- **NPU: Level Zero, not a vendor-specific NPU SDK.** Level Zero is oneAPI's real, public, vendor-neutral low-level API; Intel's NPU driver stack exposes itself through it, using the real `ZE_INIT_FLAG_VPU_ONLY` flag ("VPU" is Level Zero's own internal name for NPU-class devices) to scope discovery away from GPUs. This also means the same loader library and calling convention this crate establishes would extend naturally to a future Level-Zero-based GPU Provider, unlike a narrower NPU-only SDK would.
- **TPU: the real local Coral Edge TPU, not Google Cloud TPU.** Cloud TPU is a remote service accessed via XLA/PJRT over a network API, not a local device a `Provider::devices()` call could ever enumerate -- treating it as this crate's scope would misrepresent what "device discovery" means here. Coral's `libedgetpu` is the real, local, purchasable hardware match for a device-discovery Provider.
- **Every real signature was fetched from the actual public header, not reconstructed from memory or a documentation summary.** `ze_api.h` (`oneapi-src/level-zero`) and `edgetpu_c.h` (`google-coral/libedgetpu`) were fetched directly; `zeInit`/`zeDriverGet`'s exact parameter types, `ZE_INIT_FLAG_VPU_ONLY`'s exact bit value, and `edgetpu_device`'s exact struct layout all come from that real source, the same rigor `providers/wgpu`'s own API-fidelity work established for `wgpu` 30.0.1 by reading the vendored crate source rather than guessing.
- **`libloading`, not a vendor binding crate, for both.** Matches `providers/rocm`'s own precedent and rationale: the real API surface each baseline actually calls is a handful of stable, well-documented entry points, small and stable enough to declare directly rather than pull in a binding crate of uncertain maintenance status.

## Risks / Trade-offs

- **Both crates' real API correctness remains entirely unverified against real hardware** -- the FFI signatures are believed correct against their real, public, fetched headers, but no Level Zero NPU driver or Coral Edge TPU has ever actually executed this code.
- **`ZE_INIT_FLAG_VPU_ONLY`'s real behavior on an actual NPU-bearing machine is unverified** -- it is the correct flag per the real header, but whether a real Intel NPU driver responds to it exactly as documented can only be confirmed by a contributor with that hardware.
