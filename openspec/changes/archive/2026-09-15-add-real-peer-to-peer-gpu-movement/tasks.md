## 1. Research

- [x] 1.1 Confirmed via web search and real vendored `cudarc-0.19.9` source reading that `cudarc`'s `result.rs` wraps `cuMemcpyPeerAsync` but has no safe wrapper for `cuDeviceCanAccessPeer`/`cuCtxEnablePeerAccess` anywhere.
- [x] 1.2 Found the real, public `sys::cuDeviceCanAccessPeer`/`sys::cuCtxEnablePeerAccess` FFI declarations and the real, public `sys::CUresult::result()` error-conversion method `cudarc` uses internally.
- [x] 1.3 Found the real, public `CudaContext::cu_device()`/`cu_ctx()` accessors (needed since `CudaContext`'s own raw handle fields are `pub(crate)` to `cudarc`, not accessible externally).
- [x] 1.4 Read `cudarc`'s own `core.rs` `memcpy_dtod`/`clone_dtod` implementation and its own `peer_transfer_contexts` test: confirmed the existing `CudaKernels::clone_buffer` (via `CudaStream::memcpy_dtod`) already automatically takes the real peer-copy path whenever source and destination belong to different real contexts -- no new low-level copy primitive needed.

## 2. `providers/cuda`

- [x] 2.1 `src/peer.rs`: `device_can_access_peer` (real, explicit `cuDeviceCanAccessPeer` query) and `enable_peer_access` (real `cuCtxEnablePeerAccess`, idempotent against `CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED`).
- [x] 2.2 `CudaExecutor::copy_tensor_from_peer_admitted`: reads a peer executor's own stored buffer, calls `CudaKernels::clone_buffer` (the real cross-context peer path), admits the result in this executor's own Memory Manager bookkeeping -- mirrors `copy_tensor_admitted`'s existing structure exactly.
- [x] 2.3 `CudaProvider::executor()`: real, public accessor for the concrete `CudaExecutor`, mirroring `providers/cpu`'s `ReferenceCpuProvider::executor()`.
- [x] 2.4 `cargo test`/`cargo fmt`/`cargo clippy --all-targets -- -D warnings` all clean; all 48 pre-existing tests (1 intentionally ignored) still pass on this development workstation's single real GPU.

## 3. `integration-tests/multi-device-cpu-cuda`

- [x] 3.1 `two_real_cuda_gpus_move_a_tensor_via_real_peer_to_peer_copy_not_host_staging`: real, explicit peer-capability query, real enable, real cross-GPU device-to-device copy, verified byte-identical against the source tensor.
- [x] 3.2 Gracefully skips on any host with fewer than two real CUDA devices or no usable real peer path (never assumed) -- confirmed genuinely skipping on this development workstation (one real GPU).
- [x] 3.3 Ties the real movement to `multi_device_placement`'s own `StageMovementEdge`/`HostStagingPolicy::Forbid` (the first real use of `Forbid`, not `Permit`, in this file -- honestly reflecting that this movement genuinely never touched host memory).
- [x] 3.4 Updated the module's own top-of-file doc comment: the prior "CUDA peer access APIs are not implemented anywhere in this repository" claim no longer held.
- [x] 3.5 `cargo test`/`cargo fmt`/`cargo clippy --all-targets -- -D warnings` all clean.

## 4. Verification

- [x] 4.1 Real CI green (`quality.yml`) on GitHub's GPU-less runners, confirming graceful skip behavior there.
- [x] 4.2 Verified genuinely executing (not skipped) on the real two-GPU `arc-gpu-magnetar` CI node via `gpu-runner-smoke.yml` (`34993763826`): real `cuDeviceCanAccessPeer` returned true, real peer access enabled, real cross-GPU copy succeeded with a byte-identical result -- no "skipping" message for this test in the real job log.

## 5. Documentation

- [x] 5.1 `openspec validate add-real-peer-to-peer-gpu-movement --strict` passes.
- [x] 5.2 `SUBMODULES.md` updated: `providers/cuda` pin and compatibility-matrix row.
- [x] 5.3 README.md's top-level scope-charter status updated once archived.
