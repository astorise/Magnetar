## 1. Implementation

- [x] 1.1 `QwenSegmentBoundaryInput` (`Host(HostTensor) | Resident { resource_id, shape }`) added to `magnetar-runtime`.
- [x] 1.2 `execute_qwen_graph_with_resident_input` added, generalizing `execute_qwen_graph`/`execute_qwen_graph_nodes` with a `resident_bindings` parameter; `execute_qwen_graph` becomes a thin wrapper, signature/behavior unchanged for every pre-existing call site.
- [x] 1.3 `run_first_native_graph_segment_dispatch`'s `boundary_input` generalized to `Option<QwenSegmentBoundaryInput>`; `run_first_native_graph_segment_with_provider_and_weights`'s own public `Option<HostTensor>` signature kept unchanged, mapping internally to `QwenSegmentBoundaryInput::Host`.

## 2. Real hardware verification

- [x] 2.1 New real test `two_real_cuda_gpus_run_one_real_qwen_forward_pass_with_real_peer_to_peer_boundary_movement`: the same two-segment split, boundary hidden state moved via real `CudaExecutor::copy_tensor_from_peer_admitted`, matching the full graph within tolerance.
- [x] 2.2 Gracefully skips on any host without two real, peer-capable CUDA devices.
- [x] 2.3 Verified genuinely executing (no skip) and passing on the real `arc-gpu-magnetar` two-GPU CI node.

## 3. Cleanup and documentation

- [x] 3.1 `integration-tests/production-loading/src/tests_real_checkpoint_multi_gpu_segment.rs`'s one affected call site updated for the type change, kept on the Host-staged path.
- [x] 3.2 `tests_multi_device_cpu_cuda.rs`'s module doc comment updated: describes the new seventh test, removes the now-stale "not yet used for this specific boundary tensor" and "replacing the sixth test's Host round trip" future-work phrasing.
- [x] 3.3 `cargo test`/`cargo fmt`/`cargo clippy --all-targets -- -D warnings`/`cargo doc -D warnings` all clean; real CI green.
- [x] 3.4 `openspec validate add-real-peer-to-peer-segment-boundary-movement --strict` passes.
- [x] 3.5 README.md's scope-charter paragraph updated to describe the real, hardware-verified closure.
