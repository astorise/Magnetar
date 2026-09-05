## 1. `TensorValue` error channel in the Provider Execution API

- [x] 1.1 Add `TensorValueAdmissionError` (`Memory(MemoryError)` /
      `Provider(ProviderExecutionError)`) to `magnetar-runtime/src/provider.rs`.
- [x] 1.2 Change `ProviderExecutionApi::write_tensor_value` to
      `fn write_tensor_value(&self, id: TensorResourceId, value: TensorValue) -> Result<(), ProviderExecutionError>`,
      default `Ok(())`.
- [x] 1.3 Change `ProviderExecutionApi::write_tensor_value_admitted` to return
      `Result<(), TensorValueAdmissionError>`.
      Design revision made during implementation: the default implementation
      stays fail-closed (matching `write_tensor_admitted`'s existing
      default) rather than attempting a generic admit-then-write — a
      trait-level default has no way to know its own Provider identity for
      `MemoryPlacement::ProviderOwnedOpaque`, so it cannot safely admit on
      an arbitrary implementor's behalf. Each concrete Provider
      (`ReferenceCpuExecutor`, and `providers/cpu`/`providers/cuda`'s
      equivalents) implements the real admit-then-write-with-rollback shape
      itself, where it does know its own identity.
- [x] 1.4 Update `ReferenceCpuExecutor::write_tensor_value`
      (`reference_cpu.rs`) to the new signature — fix the discarded-`Result`
      bug (`Ok(self.write_tensor(id, tensor))`-shaped) rather than only
      satisfying the compiler.
- [x] 1.5 Update `ReferenceCpuExecutor::write_tensor_value_admitted` to the
      new signature, preserving its existing admit-then-write behavior.
- [x] 1.6 Update every in-tree `ProviderExecutionApi` mock (`tests.rs`,
      `first_native_runtime/tests.rs` — including `FailableProviderExecutionApi`
      and `DeviceResidentOnlyExecutor`) to the new signatures.
      `FailableProviderExecutionApi` additionally gained real
      `write_tensor_value`/`write_tensor_value_admitted` overrides (reusing
      its existing `fail_write` flag) rather than inheriting the fail-closed
      default, since task 4.1's regression test needs admission to actually
      succeed so the write failure specifically is what's under test.

## 2. Fix the three real `write_tensor_value_admitted` call sites

- [x] 2.1 Update `execute_qwen_graph_nodes`'s three call sites (graph input
      bindings, pending KV resource write, node output edge — around
      first_native_runtime.rs:2984/3127/3162) to match
      `TensorValueAdmissionError`'s two variants: `Memory(e) ->
      InferenceApiError::MemoryAdmissionFailed`, `Provider(e) ->
      InferenceApiError::ProviderTensorWriteFailed` (both variants already
      existed; no new `InferenceApiError` variant was needed).

## 3. Resolve real Provider bindings instead of hardcoding Reference CPU

- [x] 3.1 Add `instance: &ModelInstanceId` to
      `WeightMaterializationTransaction::begin()`; resolve
      `runtime.model_instance(instance)?.definition().placement.provider`,
      falling back to `REFERENCE_CPU_PROVIDER_NAME` only when unset. Updated
      `materialize_model_instance_weights` to pass it through.
- [x] 3.2 Add `provider: Option<ProviderBinding>` to
      `FirstNativeExecutionKvState`; set it where `execute_qwen_graph`
      resolves its own binding. Required extending `QwenGraphExecutionOutput`
      to a 4-tuple (adding the resolved `ProviderBinding`) and introducing a
      separate `QwenGraphNodesOutput` 3-tuple type for
      `execute_qwen_graph_nodes`'s own return (it doesn't resolve the
      binding itself, only `execute_qwen_graph` does) — updated all ~20
      `execute_qwen_graph` call sites; all but 2 destructure the result with
      `Ok(_)`/discard patterns already unaffected by the added element.
- [x] 3.3 Update `KvUpdateTransaction::begin()` to resolve from that new
      state field instead of the hardcoded constant, falling back to
      `REFERENCE_CPU_PROVIDER_NAME` only when the field is unset. Threaded
      through `commit_generation_step` -> `promote_pending_kv_resources`
      (which already received `&state`) -> `begin()`.
- [x] 3.4 Applied the identical fix to `discard_pending_kv_state`'s
      hardcoded Reference CPU resolution, reading the same
      `FirstNativeExecutionKvState.provider` field.

## 4. Regression tests

- [x] 4.1 New test mirroring `stage_weight_propagates_and_rolls_back_on_provider_write_failure`:
      `write_tensor_value_admitted_propagates_and_rolls_back_on_provider_write_failure`
      asserts `TensorValueAdmissionError::Provider` is returned and the
      just-admitted `MemoryAllocation` ends `Released`, not leaked.
- [x] 4.2 New test `weight_materialization_uses_the_model_instances_bound_provider`:
      a Model Instance created with `ResourceAffinity` binding the existing
      `MockKernelProvider` test double (registered under
      `magnetar:provider/mock-kernel`, no real CUDA needed) actually
      materializes weights through that Provider's storage, verified absent
      from Reference CPU's.
- [x] 4.3 Two new tests instead of one, both narrower and more direct than
      originally planned: `kv_update_transaction_resolves_the_states_bound_provider`
      and `kv_update_transaction_falls_back_to_reference_cpu_when_unbound`
      test `KvUpdateTransaction::begin`'s resolution directly (asserting
      `transaction.provider_binding`) rather than driving a full non-CPU
      generation step end-to-end -- that would need a mock Provider
      implementing every Qwen graph Operator, not just this contract, since
      `begin` is the entire fix and everything downstream already receives
      whichever binding it resolves.
- [x] 4.4 Confirmed: full `magnetar-runtime` suite passed unchanged before
      (1186/1186) and after (1190/1190 — the 4 new tests) these changes; no
      existing test's expected outcome needed touching, only mock
      trait-method signatures, exactly as design.md predicted.

## 5. Verification

- [x] 5.1 `cargo test -p magnetar-runtime --lib` — 1190/1190 green.
      `cargo build --workspace --all-targets` (magnetar-runtime +
      magnetar-cli) also clean.
- [x] 5.2 `cargo clippy -p magnetar-runtime --all-targets -- -D warnings` and
      `cargo fmt --check` clean.
- [x] 5.3 Confirmed this change's own diff touches only
      `magnetar-runtime/src/**` — `providers/cpu`/`providers/cuda` needed
      their own separate, immediate fix (task group 6 below), not part of
      this diff.

## 6. Fix the two external submodule implementors

Upgraded from "tracked follow-up" (the original plan) to done now: both
submodules are live in this same working tree, so there was no reason to
defer a breaking-change fix whose shape was already fully known.

- [x] 6.1 Updated `providers/cpu` (`magnetar-provider-cpu`)'s
      `ReferenceCpuExecutor::write_tensor_value`/`write_tensor_value_admitted`
      to the new signatures (identical fix to task 1.4/1.5's Core copy).
      `cargo test`/`clippy`/`fmt` all clean (9/9 tests passing).
- [x] 6.2 Updated `providers/cuda` (`magnetar-provider-cuda`)'s
      `CudaExecutor::write_tensor_value`/`write_tensor_value_admitted`
      (from `implement-cuda-provider-baseline`) identically. `cargo test`/
      `clippy`/`fmt` all clean (18/18 tests passing, including both
      `ProviderConformanceProfile::ProviderCore`/`ProviderCompute` on real
      hardware).
      Both submodules remain uncommitted in their own repositories (same
      status as `implement-cuda-provider-baseline`'s own pending commit
      decision) -- this task closes the *code* gap, not the *publish/pin*
      step, which is a separate decision for whoever owns pushing to
      `Magnetar-provider-CPU`/`Magnetar-provider-CUDA` and advancing this
      repository's gitlinks.
