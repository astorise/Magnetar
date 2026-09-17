## 1. Phase A -- WIT contract and real Component

- [x] 1.1 `model-component-graph.wit` bumped `1.2.0` -> `1.3.0`: `build-prefill-graph-segment`/`build-decode-graph-segment`, purely additive.
- [x] 1.2 `components/qwen`'s `build_graph`/`build_graph_segment` implementation and real recompile (`wasm32-unknown-unknown` -> `wasm-tools component new`).
- [x] 1.3 Real bug found and fixed: tied embeddings' `alias-weight-edge(lm_head, token_embedding)` failed for a segment reaching lm-head without owning the embedding lookup; fixed by registering the `token_embedding` weight edge first.
- [x] 1.4 Every checked-in fixture copy (`magnetar-runtime/fixtures/components/`, `magnetar-cli/fixtures/`) re-pinned to the new real digest; `QWEN_REAL_COMPONENT_DIGEST` updated (and `QWEN_GRAPH_COMPONENT_DIGEST`'s own, unrelated, pre-existing value correctly left untouched after an initial mistaken edit was caught and reverted).
- [x] 1.5 Full `magnetar-runtime` suite green after the digest/version propagation (1262/1262 at the time).

## 2. Phase B.1 -- synthetic correctness proof

- [x] 2.1 `filter_manifest_for_layer_range`/`qwen_weight_name_in_layer_range`: restrict a Model Instance's weight inventory to one real decoder-layer range.
- [x] 2.2 `build_first_native_prefill_graph_segment_for_config`: the segment counterpart of the full-graph builder, through the real Qwen Component's `build-prefill-graph-segment` export.
- [x] 2.3 New unit test: a two-segment split on Reference CPU matches the one full graph, bit-for-bit, using a small synthetic 2-layer fixture.
- [x] 2.4 Real bug found and fixed: `execute_qwen_graph_nodes`'s KV-collection loop assumed Vec position == real layer number; introduced `QwenLayerKvMap` (keyed by real layer number) and rewired every consumer.
- [x] 2.5 `cargo test`/`cargo fmt`/`cargo clippy --all-targets -- -D warnings`/`cargo doc -D warnings` all clean; real CI green.

## 3. Phase B.2 -- real two-GPU prefill

- [x] 3.1 `load_first_native_segment_with_provider_and_weights`/`run_first_native_graph_segment_dispatch`/`run_first_native_graph_segment_with_provider_and_weights`: public, Provider-generic segment load/dispatch API.
- [x] 3.2 New real test in `integration-tests/multi-device-cpu-cuda`: a Qwen forward pass split across two real GPUs (layers `[0, mid)` / `[mid, num_hidden_layers)`), boundary hidden state moved via an explicit Host round trip, matching the full graph on one real GPU within tolerance.
- [x] 3.3 Gracefully skips on any host without two real CUDA devices; verified genuinely executing (not skipped) on the real `arc-gpu-magnetar` CI node.

## 4. Phase C -- real checkpoint, real multi-step decode

- [x] 4.1 `build_first_native_decode_graph_segment_for_config`: the decode counterpart of the segment graph builder.
- [x] 4.2 `load_production_qwen_instance_segment_for_provider`: a `ProductionArtifactPayloadSource`-streaming segment loader (only the segment's own weights ever read from the real checkpoint file).
- [x] 4.3 `run_first_native_graph_segment_dispatch` gained `kv_history`/`absolute_position_override`; `FirstNativeProviderRunOutcome` gained `layer_kv` (additive), so a segment loads once and dispatches repeatedly across a real multi-step generation loop.
- [x] 4.4 New, cheap, always-run Reference-CPU-only unit test (`check_two_segment_split_decode_step_matches_full_graph_decode`): a real decode step split across two segment Model Instances matches the full graph's own decode step -- closes the real coverage-ratchet gap the (necessarily `#[ignore]`d) real-checkpoint test alone left.
- [x] 4.5 New real test in `integration-tests/production-loading` (`real_public_checkpoint_multi_gpu_segment_decode_matches_full_graph_on_one_gpu`, named to match `gpu-runner-smoke.yml`'s existing `real_public_checkpoint --include-ignored` filter): the real, public Qwen2.5-0.5B-Instruct checkpoint split across two real GPUs, real 6-token greedy generation (1 prefill + 5 decode), matching `run_production_qwen_generation_for_provider` (real full graph, one real GPU) exactly.
- [x] 4.6 Real hardware failure diagnosed via temporary, reverted `providers/cuda` instrumentation (two real GPU CI round trips) and root-caused to the new test's own `argmax` sampling (flattened multi-row prefill logits instead of the last row) -- not production code. Fixed; CUDA instrumentation reverted to byte-identical.
- [x] 4.7 wasm32 target compile fixed (`load_production_qwen_instance_segment_for_provider` was missing the `wasmtime-component-engine`/`not(wasm32)` cfg gate every sibling segment function already carries).
- [x] 4.8 Real CI green on every commit in this phase, including the coverage ratchet (closed by 4.4) and the real GPU smoke workflow's full `real_public_checkpoint` suite (4/4 passing).

## 5. Documentation

- [x] 5.1 `openspec validate add-real-multi-device-model-instance-placement --strict` passes.
- [x] 5.2 README.md's top-level scope-charter status updated, closing the "multi-device execution" line of the original audit.
