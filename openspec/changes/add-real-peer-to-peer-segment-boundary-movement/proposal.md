## Why

`add-real-multi-device-model-instance-placement` closed production `ModelInstance`-level placement across two real GPUs, but deliberately left its own boundary hidden-state tensor moving through an explicit Host round trip, both in its own design doc and in `multi-device-placement`'s own spec Purpose line, naming the already-proven zero-Host-round-trip `CudaExecutor::copy_tensor_from_peer_admitted` primitive (`add-real-peer-to-peer-gpu-movement`) as real, well-scoped follow-up work, not a correctness gap. This change closes that gap for the segment boundary tensor specifically.

## What Changes

`magnetar-runtime` gains `QwenSegmentBoundaryInput` (`Host(HostTensor) | Resident { resource_id, shape }`) and `execute_qwen_graph_with_resident_input`, generalizing the existing `execute_qwen_graph`/`execute_qwen_graph_nodes` so a segment's `input.hidden_states_in` edge can be satisfied either by a Host-staged write (existing behavior, unchanged for all ~26 pre-existing call sites via a thin wrapper) or by a resource id/shape pair the caller has already written into via a real peer copy. `run_first_native_graph_segment_dispatch`'s `boundary_input` parameter is generalized from `Option<HostTensor>` to `Option<QwenSegmentBoundaryInput>`; the pre-existing convenience wrapper `run_first_native_graph_segment_with_provider_and_weights` keeps its own `Option<HostTensor>` signature unchanged, mapping internally to `QwenSegmentBoundaryInput::Host`.

A new real test in `integration-tests/multi-device-cpu-cuda` runs the same two-segment Qwen split as the prior change's own real two-GPU test, but moves the boundary hidden state via a real `CudaExecutor::copy_tensor_from_peer_admitted` Device-to-Device copy instead of a Host round trip, asserting the result matches the full, unsegmented graph within tolerance.

- **BREAKING**: none. `execute_qwen_graph`'s own signature and behavior are fully unchanged (now a thin wrapper over the new, more general function with an empty resident-bindings map). `run_first_native_graph_segment_with_provider_and_weights`'s public signature is unchanged.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `multi-device-placement`: the segment-boundary tensor between two production `ModelInstance`s can now move via real, zero-Host-round-trip peer-to-peer copy, verified on two real, physically distinct GPUs, in addition to the pre-existing Host-staged path.

## Impact

- `magnetar-runtime/src/first_native_runtime.rs`: `QwenSegmentBoundaryInput` enum, `execute_qwen_graph_with_resident_input`, updated `run_first_native_graph_segment_dispatch`.
- `integration-tests/multi-device-cpu-cuda/src/tests_multi_device_cpu_cuda.rs`: new real two-GPU peer-to-peer boundary test; module doc comment updated to describe it and remove now-stale "not yet used for this specific boundary tensor" phrasing.
- `integration-tests/production-loading/src/tests_real_checkpoint_multi_gpu_segment.rs`: one call site updated for the `QwenSegmentBoundaryInput` type change, deliberately kept on the Host-staged path (out of scope for this change).
