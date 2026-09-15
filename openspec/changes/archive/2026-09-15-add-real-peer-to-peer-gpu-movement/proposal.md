## Why

`add-real-second-gpu-cuda-provider` deliberately left "real peer-to-peer GPU-to-GPU memory access" as a stated Non-Goal, since both existing multi-Device tests move data via an explicit host round trip and CUDA peer access APIs were unimplemented anywhere in this repository. With two real, physically distinct GPUs now genuinely available and peer-capable on `arc-gpu-magnetar`'s CI node, and per the user's explicit instruction to close this and the remaining real gaps, this change implements it for real: a genuine cross-GPU device-to-device copy that never touches host memory, gated by an explicit, real peer-capability check -- directly implementing `multi-device-placement`'s own long-standing "Peer Capability Is Explicit" requirement ("Runtime SHALL not infer peer access from Device similarity"), which nothing in this repository's history had exercised against real hardware until now.

## What Changes

- `providers/cuda`: new `peer` module wrapping the real `cuDeviceCanAccessPeer`/`cuCtxEnablePeerAccess` driver entry points directly (`cudarc` itself provides no safe wrapper for either, confirmed by reading its real source -- unlike its own `memcpy_peer_async`), using the same `.result()`-based `CUresult` -> `Result` conversion `cudarc` uses internally and the real, public `CudaContext::cu_device()`/`cu_ctx()` accessors.
- `CudaExecutor::copy_tensor_from_peer_admitted`: moves a tensor directly from a peer executor's own device storage into this executor's, via a real cross-context `cuMemcpyPeerAsync` (`CudaKernels::clone_buffer`'s existing `CudaStream::memcpy_dtod` already takes this path automatically whenever source and destination belong to different real CUDA contexts -- confirmed against `cudarc`'s own `peer_transfer_contexts` test) -- never touching host memory, structurally distinct from every existing cross-Device movement in this repository.
- `CudaProvider::executor()`: exposes the concrete `CudaExecutor` (mirroring `providers/cpu`'s `ReferenceCpuProvider::executor()`), needed since `execution_api()` type-erases it behind the generic `ProviderExecutionApi` trait, which has no peer-copy method.
- `integration-tests/multi-device-cpu-cuda`: a new test, `two_real_cuda_gpus_move_a_tensor_via_real_peer_to_peer_copy_not_host_staging`, queries real peer capability explicitly, enables it, and moves a tensor between two real GPUs' own device memory -- gracefully skipping on any host with fewer than two real CUDA devices or no usable real peer path (never assumed).
- **BREAKING**: none. All new, additive API surface.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `cuda-provider`: gains real peer-access query/enable and cross-context device-to-device copy requirements.
- `multi-device-placement`: gains a requirement that real peer-to-peer movement, once confirmed available, is genuinely used rather than merely representable -- closing the "Peer Capability Is Explicit" requirement's real-hardware gap.

## Impact

- `providers/cuda/src/{peer,executor,provider,lib}.rs`: new `peer` module, `copy_tensor_from_peer_admitted`, `executor()` accessor.
- `integration-tests/multi-device-cpu-cuda/src/tests_multi_device_cpu_cuda.rs`: new test, updated module doc comment.
- `SUBMODULES.md`: `providers/cuda` pin and compatibility-matrix row updated.
