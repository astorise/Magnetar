## 1. `concat` Operator catalog entry

- [x] 1.1 Add `ShapeRule::RowConcat` to `magnetar-runtime/src/operator.rs`: both inputs share the same column count; output row count equals the sum of both inputs' row counts. Mirror `ShapeRule::RowBroadcastAdd`'s existing match-arm structure.
- [x] 1.2 Add `("concat", OperatorFamily::Tensor, 2, 1, ShapeRule::RowConcat)` to the Operator catalog.
- [x] 1.3 Add a unit test proving `ShapeRule::RowConcat` accepts a valid pair (same columns, additive rows) and rejects a column mismatch, mirroring the existing `ShapeRule::RowBroadcastAdd` tests.

## 2. Reference CPU `concat` implementation

- [x] 2.1 Implement `concat` in `magnetar-runtime/src/reference_cpu.rs` (the in-crate double): row-major byte concatenation of two same-trailing-dimension tensors into `[a_rows + b_rows, ..rest]`, dispatched via a new `"concat"` match arm (mirroring `"add"`/`"mul"`) and advertised via `baseline_advertisement("concat", OperatorFamily::Tensor)`. (Duplicated rather than sharing `first_native_runtime.rs`'s own `concat_rows` -- that one stays as the KV-append block's own removed-in-task-5 helper; this is `reference_cpu.rs`'s independent generic-Operator double, matching how `add`'s broadcast logic already exists independently in both places.)
- [x] 2.2 Implement `concat` identically in the `providers/cpu` submodule (`astorise/Magnetar-provider-CPU`), for CPU/CUDA conformance parity.
- [x] 2.3 Add tests proving `concat` produces the correct concatenated tensor and rejects a trailing-dimension mismatch, in both implementations.

## 3. CUDA `concat` implementation

- [x] 3.1 Implement `CudaKernels::concat` in `providers/cuda/src/kernels.rs`: allocate a fresh `[a_rows + b_rows, ..rest]` device buffer (`alloc_zeros`) and copy each input directly into its own non-overlapping mutable view (`CudaSlice::split_at_mut` + `CudaStream::memcpy_dtod`) -- no `.cu` kernel needed, confirming the design's prediction.
- [x] 3.2 Dispatch `"concat"` in `CudaExecutor::run_invocation`'s operator match (`providers/cuda/src/executor.rs`).
- [x] 3.3 Advertise `"concat"` in `cuda_kernel_advertisements` (`providers/cuda/src/advertisements.rs`); updated `kernel_advertisements_agree_with_availability`'s hardcoded count/name-set test (10 -> 11, `"concat"` added).
- [x] 3.4 Added `concat_matches_reference_cpu` to `tests_conformance.rs`, run and passing on real CUDA hardware (RTX 3070 Ti), byte-identical against `magnetar_provider_cpu::concat` for the same two input tensors. Full `providers/cuda` suite (34 tests, `--include-ignored`) passes on real hardware.

## 4. `copy_tensor_admitted`: host-free Tensor Resource duplication

- [x] 4.1 Add `fn copy_tensor_admitted(&self, memory: &mut MemoryManager, from: &TensorResourceId, to: TensorResourceId, class: MemoryAllocationClass, owner: MemoryAllocationOwner) -> Result<(), TensorValueAdmissionError>` to `ProviderExecutionApi` (`magnetar-runtime/src/provider.rs`), documented as duplicating `from`'s current bytes to `to` without requiring host materialization, replacing (and releasing) whatever `to` previously held. Default implementation fails closed, mirroring `write_tensor_value_admitted`'s own default.
- [x] 4.2 Implemented on Reference CPU (`reference_cpu.rs` and `providers/cpu`) as a `HostTensor` clone under the new id, admitted the same way `write_tensor_admitted` already is; wired into both crates' `ProviderExecutionApi` trait impls.
- [x] 4.3 Implemented on CUDA (`providers/cuda`) as a real device-to-device buffer copy (`CudaKernels::clone_buffer`, `cudarc`'s `clone_dtod`) under the new id, admitted the same way (mirroring `write_tensor_admitted`'s admit-then-write-with-rollback shape), replacing any existing allocation at `to`; wired into the trait impl.
- [x] 4.4 Added tests in all three implementations (magnetar-runtime, providers/cpu, providers/cuda -- the last run and passing on real CUDA hardware) proving: the copy is byte-correct (verified by downloading only the destination, never the source); a second/repeated copy to the same `to` id does not grow the active allocation count beyond the source's own plus one stable destination allocation; copying a nonexistent `from` id fails structurally (`TensorValueAdmissionError`), not silently. Full suites re-run clean: magnetar-runtime (1249+173), providers/cpu (21), providers/cuda (37, real hardware).

## 5. Wire `concat` into KV-history-append, eliminating the blocking read

- [x] 5.1 In `execute_qwen_graph_nodes`'s `GraphKvCacheBehavior::Append` block, replaced `provider.read_tensor_value(historical_resource).into_host(..)` + `concat_rows(..)` with a dispatch through the new `"concat"` Operator, pre-admitting the output directly under `output_resource_id`. **Real design correction found during implementation**: the concat step is Runtime-side bookkeeping tied to an existing node's output edge, not itself a node the static graph declares, so it has no `PreparedExecutionPlan` node binding under any id -- `dispatch_reference_cpu_operator_pre_admitted`/`_multi` (the mechanism every *real* graph node uses) cannot be reused as originally planned (confirmed by two real failure modes: a synthetic node id fails `PlanNodeBindingMissing`; reusing the real node's own id would silently resolve its *original* Kernel binding, e.g. "rope", not "concat"). Implemented instead as a new, purpose-built `dispatch_qwen_concat` performing a real, live Kernel Registry selection (mirroring `dispatch_reference_cpu_operator_multi`'s own no-Prepared-Plan branch, not `prepared_candidate_for_operation`'s synthetic-candidate path, which stays production-forbidden), using the real node's own already-resolved resource affinity so selection still correctly targets the actual bound Provider. Deliberately does not participate in the per-node causal-chain contract (`GraphNodeReady`/`PlanBindingResolved`), matching `concat_rows`'s own prior zero-observability posture -- confirmed correct by `e2e_report_contains_required_metadata_fields`/`e2e_ci_can_run_without_gpu_and_reports_only_expected_required_failure` (the "no shortcuts" conformance suite) passing.
- [x] 5.2 Confirmed `needs_explicit_edge_write = false` is set explicitly after the concat dispatch, skipping the later unconditional `into_host` at the edge-write block; the full `magnetar-runtime` test suite (1249+173 tests, including every existing E2E/KV/graph-executor test) passes unchanged.
- [x] 5.3 Replaced the pending-write round-trip with `copy_tensor_admitted(.., &id, pending_resource, ..)` when `output_tensor` is already `Resident` (decode's KV-append case, and prefill's "Output" case whenever the producing Kernel wrote directly into its edge); a genuinely `Host`-typed value (e.g. "rope", which has no Kernel-level output identity) still uses the original host-typed admitted write, since there is no device-resident source to copy from.
- [x] 5.4 Verified via real behavior rather than a source-text guard: `concat_rows` (the removed host-materialization helper) is now `#[cfg(test)]`-gated -- it compiles only into the Rust-synthesized test-oracle path (`execute_qwen_decode_hidden_states_through_dispatch`), confirmed by `cargo clippy --all-targets --all-features -- -D warnings` (dead-code) that production's own KV-history-append path no longer references it at all.
- [x] **Verified on real CUDA hardware**: `real_production_ingestion_generates_on_real_cuda_hardware` (tiny synthetic bundle) now runs a genuine 4-token decode (prefill + 3 real decode steps) on real CUDA hardware and passes. `real_public_checkpoint_multi_token_decode_matches_between_cpu_and_cuda` (the real public Qwen2.5-0.5B-Instruct checkpoint, generalized from task 12.5's prefill-only shape) confirms all 4 generated tokens and decoded text match exactly between Reference CPU and real CUDA hardware: `" 100"`.

## 6. Close the commit-path round-trip

- [ ] 6.1 In `promote_pending_kv_layer_role` (`first_native_runtime.rs:5214-5272`), replace `read_tensor(pending_resource)` + `write_tensor(committed_resource, ..)` with `copy_tensor_admitted(.., &pending_resource, committed_resource, ..)`.
- [ ] 6.2 Fix the `MemoryPlacement` inconsistency: record `MemoryPlacement::Device(device_binding)` for a Provider that reports a concrete Device (CUDA), keeping `ProviderOwnedOpaque` only when no concrete Device identity exists.
- [ ] 6.3 Add a test proving the commit step no longer calls the host-typed `read_tensor`/`write_tensor` for this path (source guard or behavioral, whichever the existing test suite's own convention favors here).

## 7. CUDA genuinely supports multi-step decode

- [x] 7.1 Removed `CudaProvider::supports_multi_step_decode`'s `false` override (`providers/cuda/src/provider.rs`), reverting to the trait's `true` default -- doc comment explains what now makes this true.
- [x] 7.2 Removed `close-tachyon-scope-audit-gaps`'s `multi_step_decode_on_real_cuda_hardware_fails_with_explicit_unsupported` test entirely (it asserted CUDA *fails* multi-step decode, no longer true); the generic gate remains proven against a synthetic Provider in `tests_production_loading_e2e.rs`, unaffected.
- [x] 7.3 Generalized `tests_production_loading_cuda_e2e.rs`'s `real_production_ingestion_generates_on_real_cuda_hardware` from `max_tokens: 1` to `max_tokens: 4` (3 real decode steps beyond prefill), asserting exactly 4 generated tokens. **Passes on real CUDA hardware.**
- [x] 7.4 Generalized and renamed `tests_real_checkpoint_smoke.rs`'s `real_public_checkpoint_prefill_output_matches_between_cpu_and_cuda` to `real_public_checkpoint_multi_token_decode_matches_between_cpu_and_cuda` (`max_tokens: 4`), asserting the full generated token sequence and decoded text match exactly between Reference CPU and CUDA. **Passes on the real public Qwen2.5-0.5B-Instruct checkpoint on real CUDA hardware**: both Providers generated the identical 4-token sequence, decoding to `" 100"`.

## 8. Bounded memory growth verification

- [ ] 8.1 Add a real-hardware test that decodes a real multi-token sequence on CUDA and asserts the Memory Manager's active allocation count for this session's KV resources is the same after the last step as after the first (one live allocation per layer per role per KV identity), not growing with step count -- mirroring `close-tachyon-scope-audit-gaps`'s `unloading_a_real_production_instance_leaves_no_memory_manager_allocation` active-vs-ever-allocated pattern.
- [ ] 8.2 Run this test via `gpu-runner-smoke.yml` alongside the existing real-hardware suite.

## 9. Documentation

- [ ] 9.1 Update `README.md`'s "Qwen production loading" section: move multi-step CUDA decode from "explicitly not yet supported" to the supported list, describing the real device-resident mechanism (briefly) and the real verified evidence (CPU/CUDA parity, bounded memory growth).
- [ ] 9.2 Update `docs/production-model-loading-integration.md`'s "Current profile and limits" section accordingly; note that `Provider::supports_multi_step_decode`/`InferenceApiError::Unsupported` remain real, generic mechanisms for a Provider that genuinely cannot support a request shape -- CUDA no longer being an example of one, but the mechanism itself is unchanged and still real for a future Provider that needs it.
- [ ] 9.3 Document `ProviderExecutionApi::copy_tensor_admitted` (doc comment on the trait method) for external Provider implementers.
- [ ] 9.4 Update `SUBMODULES.md`'s `providers/cuda`/`providers/cpu` rows to mention the new `concat` Operator and `copy_tensor_admitted` implementations.

## 10. Quality gates

- [ ] 10.1 Run `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo doc --no-deps --all-features`, `cargo test --all-features` for `magnetar-runtime` (default features and `--features wasmtime-component-engine`), `providers/cpu`, `providers/cuda` (including real-hardware `--include-ignored` tests), and `magnetar-cli` -- all clean.
- [ ] 10.2 Re-run the full `integration-tests/production-loading` suite, including the real public checkpoint tests, confirming real multi-token CPU/CUDA parity and no regression elsewhere.
- [ ] 10.3 Run `gpu-runner-smoke.yml` on real hardware and confirm the new/generalized multi-token decode and bounded-memory-growth tests pass.
- [ ] 10.4 Run `openspec validate --all --strict` and archive only after every task above is complete with linked evidence (commit SHA, `Quality` and `GPU Runner Smoke Test` run IDs on that exact SHA).
