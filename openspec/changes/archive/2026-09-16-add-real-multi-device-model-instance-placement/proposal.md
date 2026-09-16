## Why

`multi-device-placement`'s own spec has stated this gap explicitly since `add-real-device-loss-degraded-replan-state-machine`: "Full production `ModelInstance`-level placement across more than one real GPU remains unimplemented -- `ModelInstancePlacement` still structurally binds one Provider/Device per instance." This is the last of the four real gaps identified after `add-real-second-gpu-cuda-provider`, and the last item from the original scope-charter audit's "multi-device execution, additional Providers" line. `ModelInstancePlacement` still binds one Provider/Device per instance today -- this change does not restructure that. Instead it closes the gap the "safe" design (chosen explicitly over invasively touching `ctx.provider` at 30+ call sites in `first_native_runtime.rs`) always intended: two separate, ordinary `ModelInstance`s, each on its own real Device, each owning one real decoder-layer range, with the boundary tensor between them moved explicitly.

## What Changes

**Phase A -- WIT contract.** `model-component-graph.wit` bumped `1.2.0` -> `1.3.0` (purely additive): `build-prefill-graph-segment`/`build-decode-graph-segment` export decoder layer range `[start_layer, end_layer)` instead of the whole stack, starting from `token_ids`+embedding (`start_layer == 0`) or a `hidden_states_in` input otherwise, ending at the real lm-head (`end_layer == num_hidden_layers`) or a raw hidden-state output otherwise. The real, checked-in `components/qwen` Component recompiled against it; a real Component-side bug this surfaced (tied-embeddings' `alias-weight-edge` failing for a segment that reaches lm-head without also owning the embedding lookup) fixed at the source.

**Phase B.1 -- synthetic correctness proof.** `magnetar-runtime` gains `filter_manifest_for_layer_range`/`qwen_weight_name_in_layer_range` (restrict a Model Instance's weight inventory to one layer range) and `build_first_native_prefill_graph_segment_for_config` (the segment counterpart of the existing full-graph builder). A new unit test proves, bit-for-bit, that splitting a forward pass into two segment `ModelInstance`s on Reference CPU matches the one full graph. This also fixed a real KV-collection bug the segment work exposed: `execute_qwen_graph_nodes`'s final K/V loop assumed Vec-position always equaled real layer number (true only for a graph touching every layer) -- now a `QwenLayerKvMap` (`BTreeMap<usize, FirstNativeLayerKvState>`) keyed by real layer number throughout, including the KV commit/promotion path.

**Phase B.2 -- real two-GPU prefill.** New public, Provider-generic API (`load_first_native_segment_with_provider_and_weights`, `run_first_native_graph_segment_dispatch`, `run_first_native_graph_segment_with_provider_and_weights`) lets a caller load a segment `ModelInstance` against an arbitrary registered Provider and dispatch its graph. A real test in `integration-tests/multi-device-cpu-cuda` splits a Qwen forward pass across two real GPUs, the boundary hidden state crossing through an explicit Host round trip, matching the full graph on one real GPU.

**Phase C -- real checkpoint, real multi-step decode.** `build_first_native_decode_graph_segment_for_config` (the decode counterpart of the segment graph builder) and `load_production_qwen_instance_segment_for_provider` (a `ProductionArtifactPayloadSource`-streaming segment loader -- only the segment's own weights are ever read from the real Safetensors file). `run_first_native_graph_segment_dispatch` gained `kv_history`/`absolute_position_override` so a segment can be loaded once and dispatched repeatedly (prefill, then every decode step), threading its own updated `layer_kv` forward each time. A real test in `integration-tests/production-loading`, using the same real, public Qwen2.5-0.5B-Instruct checkpoint `tests_real_checkpoint_smoke.rs` already verifies, splits it across two real GPUs and runs a real 6-token greedy generation loop (1 prefill + 5 decode steps), matching `run_production_qwen_generation_for_provider` (the real full graph on one real GPU) exactly.

- **Cross-Device movement stays an explicit Host round trip**, not yet the real, zero-Host-round-trip `CudaExecutor::copy_tensor_from_peer_admitted` primitive `add-real-peer-to-peer-gpu-movement` already proved possible -- a real performance optimization for later, not a correctness gap this change's own proofs depend on.
- **BREAKING**: none. `FirstNativeProviderRunOutcome` gained a `layer_kv` field (additive); every pre-existing constructor updated, no existing caller's behavior changed.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `multi-device-placement`: gains the real, hardware-verified proof that production `ModelInstance`-level placement across more than one real GPU is possible via two separate Model Instances + explicit movement, closing the Purpose line's own stated gap.
- `model-component-graph-contract`: gains the `1.3.0` segment-graph-building capability (`build-prefill-graph-segment`/`build-decode-graph-segment`).

## Impact

- `magnetar-runtime/wit/model-component-graph.wit`, `components/qwen/wit/model-component-graph.wit`, `components/qwen/src/lib.rs` (submodule, real recompile).
- `magnetar-runtime/src/first_native_runtime.rs`: new segment-graph-building, segment-loading (fixture and production), segment-dispatch, and `QwenLayerKvMap` functions/types; two new unit tests.
- `integration-tests/multi-device-cpu-cuda/src/tests_multi_device_cpu_cuda.rs`: new real two-GPU segment test.
- `integration-tests/production-loading/src/tests_real_checkpoint_multi_gpu_segment.rs`: new real checkpoint, real two-GPU, real multi-step decode test.
- `magnetar-cli/fixtures/`, `magnetar-runtime/fixtures/components/`: real recompiled Qwen Component artifact + manifest, re-pinned.
