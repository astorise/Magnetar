## 1. `concat` Operator catalog entry

- [ ] 1.1 Add `ShapeRule::RowConcat` to `magnetar-runtime/src/operator.rs`: both inputs share the same column count; output row count equals the sum of both inputs' row counts. Mirror `ShapeRule::RowBroadcastAdd`'s existing match-arm structure.
- [ ] 1.2 Add `("concat", OperatorFamily::Tensor, 2, 1, ShapeRule::RowConcat)` to the Operator catalog.
- [ ] 1.3 Add a unit test proving `ShapeRule::RowConcat` accepts a valid pair (same columns, additive rows) and rejects a column mismatch, mirroring the existing `ShapeRule::RowBroadcastAdd` tests.

## 2. Reference CPU `concat` implementation

- [ ] 2.1 Implement `concat` in `magnetar-runtime/src/reference_cpu.rs` (the in-crate double): row-major byte concatenation of two same-column tensors into `[a_rows + b_rows, cols]`, reusing the existing `concat_rows` logic in `first_native_runtime.rs` (extract it to a shared location if both call sites need it, or duplicate the ~10-line function -- match whatever `add`'s broadcast logic did when it needed the same logic in two places this session).
- [ ] 2.2 Implement `concat` identically in the `providers/cpu` submodule (`astorise/Magnetar-provider-CPU`), for CPU/CUDA conformance parity.
- [ ] 2.3 Add tests proving `concat` produces the correct concatenated tensor and rejects a column-count mismatch, in both implementations.

## 3. CUDA `concat` implementation

- [ ] 3.1 Implement `CudaKernels::concat` in `providers/cuda/src/kernels.rs`: allocate a fresh `[a_rows + b_rows, cols]` device buffer (`alloc_zeros`) and copy each input's bytes into it at the correct offset via device-to-device copies (the same `clone_dtod`-class primitive `rope` already uses at `kernels.rs:388`); validate column-count equality before allocating.
- [ ] 3.2 Dispatch `"concat"` in `CudaExecutor::run_invocation`'s operator match (`providers/cuda/src/executor.rs`).
- [ ] 3.3 Advertise `"concat"` in `cuda_kernel_advertisements` (`providers/cuda/src/advertisements.rs`).
- [ ] 3.4 Add `concat_matches_reference_cpu` to `tests_conformance.rs`, run on real CUDA hardware, proving byte-identical output against Reference CPU for the same two input tensors.

## 4. `copy_tensor_admitted`: host-free Tensor Resource duplication

- [ ] 4.1 Add `fn copy_tensor_admitted(&self, memory: &mut MemoryManager, from: &TensorResourceId, to: TensorResourceId, class: MemoryAllocationClass, owner: MemoryAllocationOwner) -> Result<(), TensorValueAdmissionError>` to `ProviderExecutionApi` (`magnetar-runtime/src/provider.rs`), documented as duplicating `from`'s current bytes to `to` without requiring host materialization, replacing (and releasing) whatever `to` previously held.
- [ ] 4.2 Implement it on Reference CPU (`reference_cpu.rs` and `providers/cpu`) as a `HostTensor` clone under the new id, admitted the same way `write_tensor_value_admitted` already is.
- [ ] 4.3 Implement it on CUDA (`providers/cuda`) as a device-to-device buffer copy under the new id, admitted the same way, replacing any existing allocation at `to`.
- [ ] 4.4 Add tests (both implementations) proving: the copy is byte-correct; a second copy to the same `to` id releases the first allocation (active allocation count does not grow); copying a nonexistent `from` id fails structurally, not silently.

## 5. Wire `concat` into KV-history-append, eliminating the blocking read

- [ ] 5.1 In `execute_qwen_graph_nodes`'s `GraphKvCacheBehavior::Append` block (`first_native_runtime.rs:3811-3864`), replace `provider.read_tensor_value(historical_resource).into_host(..)` + `concat_rows(..)` with a dispatch through the new `"concat"` Operator via the same pre-admitted-output-target mechanism `dispatch_qwen_binary_same_shape` already uses for "add"/"mul", targeting `output_resource_id` directly so the result lands device-resident under the edge's own identity.
- [ ] 5.2 Confirm `needs_explicit_edge_write` is `false` for this case afterward (the concat's own output already satisfies `output_resource_id`), and that the unconditional `output_tensor.into_host(..)` at the edge-write block (`:3941-3968`) is skipped for it -- add a regression test (or extend the existing source-guard pattern) proving this.
- [ ] 5.3 Replace the pending-write round-trip (`:3883-3916`: `into_host` + `write_tensor_value_admitted(TensorValue::Host(..))`) with `copy_tensor_admitted(.., &output_resource_id, pending_resource, ..)`.
- [ ] 5.4 Add/extend the static source guard in `first_native_runtime/tests.rs` (mirroring `check_execute_qwen_graph_nodes_transport_has_no_host_tensor_typed_calls`) to confirm the KV-append block no longer contains `.into_host(` or `.read_tensor_value(` calls for the historical-read path.

## 6. Close the commit-path round-trip

- [ ] 6.1 In `promote_pending_kv_layer_role` (`first_native_runtime.rs:5214-5272`), replace `read_tensor(pending_resource)` + `write_tensor(committed_resource, ..)` with `copy_tensor_admitted(.., &pending_resource, committed_resource, ..)`.
- [ ] 6.2 Fix the `MemoryPlacement` inconsistency: record `MemoryPlacement::Device(device_binding)` for a Provider that reports a concrete Device (CUDA), keeping `ProviderOwnedOpaque` only when no concrete Device identity exists.
- [ ] 6.3 Add a test proving the commit step no longer calls the host-typed `read_tensor`/`write_tensor` for this path (source guard or behavioral, whichever the existing test suite's own convention favors here).

## 7. CUDA genuinely supports multi-step decode

- [ ] 7.1 Remove `CudaProvider::supports_multi_step_decode`'s `false` override (`providers/cuda/src/provider.rs`), reverting to the trait's `true` default -- update its doc comment to describe what now makes this true instead of false.
- [ ] 7.2 Retarget `close-tachyon-scope-audit-gaps`'s `multi_step_decode_on_real_cuda_hardware_fails_with_explicit_unsupported` test: either remove it (the generic gate is already proven against a synthetic Provider in `tests_production_loading_e2e.rs`) or repurpose it into a real-hardware test proving CUDA *succeeds* now, whichever the actual state of the generic-gate test coverage warrants once this task starts.
- [ ] 7.3 Generalize `tests_production_loading_cuda_e2e.rs`'s `real_production_ingestion_generates_on_real_cuda_hardware` from `max_tokens: 1` (prefill-only) to a real multi-token decode (e.g. 8 tokens, matching the audit's own "8/16+ generated tokens" bar), updating its doc comment to no longer describe the now-closed gap as a limitation.
- [ ] 7.4 Generalize `tests_real_checkpoint_smoke.rs`'s `real_public_checkpoint_prefill_output_matches_between_cpu_and_cuda` from prefill-only to real multi-token decode, asserting the full generated token sequence (not just one token) matches exactly between Reference CPU and CUDA on the real public checkpoint.

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
