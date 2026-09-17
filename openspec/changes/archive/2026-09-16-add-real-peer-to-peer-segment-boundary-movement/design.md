## Context

`add-real-multi-device-model-instance-placement` proved a production Qwen forward pass can split across two real GPUs via two separate segment `ModelInstance`s, with the boundary hidden-state tensor moved through an explicit Host round trip. `add-real-peer-to-peer-gpu-movement`, from an earlier sub-chantier, already proved `CudaExecutor::copy_tensor_from_peer_admitted` can move an arbitrary Tensor Resource between two real, peer-capable GPUs without touching host memory -- but never wired that primitive into the segment boundary specifically. This change closes that gap.

## Goals / Non-Goals

**Goals:**
- Let the segment boundary tensor move via a real, zero-Host-round-trip Device-to-Device copy when the caller has already established peer access.
- Keep every pre-existing caller of `execute_qwen_graph`/`run_first_native_graph_segment_dispatch`/`run_first_native_graph_segment_with_provider_and_weights` fully unaffected.

**Non-Goals:**
- Making peer-to-peer the default or only path -- Host staging remains available and unchanged for callers without peer access (or for CPU-involved segments, where peer access does not apply).
- Converting `tests_real_checkpoint_multi_gpu_segment.rs`'s own real-checkpoint decode loop to peer-to-peer -- left as further, smaller follow-up work.
- Any change to `ModelInstancePlacement`'s own structure.

## Decisions

- **A new enum (`QwenSegmentBoundaryInput`), not a new function.** `execute_qwen_graph` keeps its exact existing signature via a thin wrapper delegating to the new, more general `execute_qwen_graph_with_resident_input` with an empty resident-bindings map -- zero behavior change for ~26 existing call sites.
- **The caller performs the peer copy, not the dispatch function.** `run_first_native_graph_segment_dispatch` accepts an already-Device-resident `(resource_id, shape)` pair rather than performing the copy itself, keeping Provider-specific peer-access setup (capability query, `cuCtxEnablePeerAccess`) at the call site where the two real Executors are already available, matching the existing standalone peer-copy test's own pattern.

## Risks / Trade-offs

- **Peer access must be established by the caller before dispatch.** If a caller passes `QwenSegmentBoundaryInput::Resident` without having actually written real data into that resource id (e.g., skipped the peer copy), dispatch would read whatever garbage or absent resource is there. This is the same caller-responsibility shape as the pre-existing standalone peer-copy primitive already has; not a new risk class introduced by this change.
