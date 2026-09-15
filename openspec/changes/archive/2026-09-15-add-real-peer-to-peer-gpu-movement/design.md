## Context

`multi-device-placement`'s own spec has carried a "Peer Capability Is Explicit" requirement since before this session began ("Runtime SHALL not infer peer access from Device similarity ... Given two GPUs have no usable peer path ... zero-copy peer placement is rejected"), but nothing in this repository's history had ever driven that requirement against real hardware -- the two prior multi-Device tests both move data via an explicit host round trip by design, and `cudarc` (the CUDA Provider's own dependency) exposes no safe wrapper for either real peer-access driver call at all. This change closes that gap now that real, peer-capable GPU hardware is available in CI.

## Goals / Non-Goals

**Goals:**
- A real, explicit peer-capability query (`cuDeviceCanAccessPeer`) that never assumes access from Device similarity, even between two identical real GPUs.
- A real cross-GPU device-to-device copy that provably never touches host memory (structurally: no `HostTensor` materializes anywhere in the call path).
- Verified genuinely executing (not skipped) on real, physically distinct, peer-capable GPU hardware.

**Non-Goals:**
- Wiring peer-copy into the generic `KernelSelectionRequest`/`KernelDispatchPlan` dispatch contract, or into `MultiDevicePlacementPlan`'s own movement-edge selection logic as an automatic, policy-chosen path -- this change adds the real, callable primitive and a real, direct test of it; automatic selection between peer-copy and host-staged movement based on real runtime conditions remains a separate, future increment.
- Peer access topology beyond a direct two-GPU pair (e.g. NVLink switch fabrics, more than two GPUs) -- the real hardware available is exactly two GPUs on one PCIe/NVLink domain.
- Disabling peer access (`cuCtxDisablePeerAccess`) or any peer-access lifecycle management beyond enabling it once -- out of scope for a first real proof.

## Decisions

- **Call the real `sys::` driver functions directly, not add a dependency or vendor a patch to `cudarc`.** `cudarc`'s own `result.rs` has no `peer_access`/`can_access` entries anywhere (confirmed by reading its real source, not assumed) -- but it does export the raw `sys` module publicly, including both real functions and the identical `.result()`-based error-conversion convention its own internal code uses. Matching that exact convention, rather than introducing a new pattern or a fork of `cudarc`, keeps this addition consistent with how the rest of `providers/cuda` already talks to `cudarc`.
- **`enable_peer_access` never checks or infers capability itself -- callers must call `device_can_access_peer` first.** A version that silently checked-then-enabled internally was considered and rejected: it would make "explicit" a property of the module's own internal control flow rather than something the caller visibly does, undermining the exact "not inferred, not assumed" posture `multi-device-placement`'s own requirement demands. The real test in `integration-tests/multi-device-cpu-cuda` calls both steps separately and visibly.
- **Reuse `CudaKernels::clone_buffer` rather than add a new low-level copy primitive.** Investigation (reading `cudarc`'s real `core.rs`) found that `CudaStream::memcpy_dtod` -- which `clone_buffer` already calls -- automatically detects when source and destination belong to different real CUDA contexts and takes the real `cuMemcpyPeerAsync` path with no special handling required from the caller; `clone_buffer` therefore already does exactly what a peer copy needs once called across two executors' own contexts. No new low-level CUDA copy code was written.
- **`CudaProvider::executor()` exposes the concrete type rather than adding peer-copy to the generic `ProviderExecutionApi` trait.** Adding it to the trait would force every other Provider (CPU, and any future one) to implement or explicitly reject a method that is fundamentally CUDA-context-specific; exposing the concrete `CudaExecutor` (mirroring `providers/cpu`'s own `ReferenceCpuProvider::executor()` precedent) keeps the generic trait unchanged while still giving a caller that specifically knows it has two `CudaProvider`s real, typed access to the real method.

## Risks / Trade-offs

- **`cuMemcpyPeerAsync` is documented to work even without `cuCtxEnablePeerAccess` having been called** (confirmed by reading `cudarc`'s own `peer_transfer_contexts` test, which calls the cross-context copy with no prior enable step) -- meaning this change's own explicit enable step, while real and verified, is not strictly load-bearing for the copy itself to succeed on this hardware. It remains real and correct to call regardless: the explicit query-then-enable sequence is what `multi-device-placement`'s own requirement asks for, independent of whether the underlying driver would have silently permitted the copy anyway, and enabling it is a real prerequisite for other peer-access patterns (e.g. direct kernel-level cross-GPU memory access) this change does not itself exercise.
- **Only one real peer topology (two identical GPUs on one host) has been verified** -- multi-GPU-switch or non-peer-capable topologies remain unverified.
