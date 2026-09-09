## Context

`magnetar-runtime/src/first_native_runtime.rs` (11,364 lines) implements Magnetar's first real, working inference path: a minimal Qwen-shaped fixture graph driven end-to-end through the actual `Runtime` API. It is not test-only scaffolding — `materialize_model_instance_weights` and the `E2eRuntimeModelExecutionEngine`'s generation loop (`commit_generation_step` → `promote_pending_kv_resources`) are real, non-`#[cfg(test)]` production code, re-exported from the crate root.

Two internal transactions in that file — `WeightMaterializationTransaction::begin()` and `KvUpdateTransaction::begin()` — each construct `ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)` directly rather than reading the Provider binding Runtime already resolved for the call. This was confirmed by direct code reading (not just the audit that first flagged it):

- `Runtime::create_model_instance` (runtime.rs:601–634) already cross-validates the caller's `ResourceAffinity` against `loaded.plan().provider_binding`/`.device_binding` and stores the result in `ModelInstanceDefinition.placement: ModelInstancePlacement { provider: Option<ProviderBinding>, device: Option<DeviceBinding>, .. }` (model_instance.rs:557–572). `WeightMaterializationTransaction::begin()` is one `runtime.model_instance(instance)` call away from this already-resolved value and ignores it.
- `execute_qwen_graph` (first_native_runtime.rs:2867) *already does this correctly* for graph dispatch: `prepared_plan.node_bindings.first().map(|binding| binding.provider.clone())`. `KvUpdateTransaction::begin()`, three call frames away in the same file's generation loop, does not reuse that mechanism and hardcodes CPU instead — so a KV resource written under one resolved Provider binding by graph execution can be looked for under a *different, wrong* binding at commit time.

Separately, `ProviderExecutionApi`'s `TensorValue` mutation methods (`provider.rs`) have an inconsistent error channel. `write_tensor` (the `HostTensor`-typed sibling) already returns `Result<(), ProviderExecutionError>`, and `WeightMaterializationTransaction::stage_weight` (first_native_runtime.rs:4964–5049) already establishes the correct pattern for it: admit through `MemoryManager::allocate` (`MemoryError` channel), then call `executor.write_tensor(...)` (`ProviderExecutionError` channel), and on write failure explicitly release the allocation just admitted before propagating `InferenceApiError::ProviderTensorWriteFailed`. This exact pattern is regression-tested (`stage_weight_propagates_and_rolls_back_on_provider_write_failure`, first_native_runtime/tests.rs:722–764).

The `TensorValue`-typed pair never got this treatment: `write_tensor_value` returns bare `()` (no channel at all — confirmed dead in production: `execute_qwen_graph_nodes` never calls it directly), and `write_tensor_value_admitted` returns `Result<(), MemoryError>` only, used at exactly three real call sites (first_native_runtime.rs:2984, 3127, 3162), all inside `execute_qwen_graph_nodes`, all mapping any error to `InferenceApiError::MemoryAdmissionFailed` today — because that is genuinely the only failure category the current signature can express, not because a Provider-native write failure can't happen.

## Goals / Non-Goals

**Goals:**
- `WeightMaterializationTransaction`/`KvUpdateTransaction` resolve the Provider binding actually associated with the Model Instance/prepared plan, so a registered non-CPU Provider becomes reachable by both, with Reference CPU's existing behavior preserved exactly when no other Provider was resolved (today's only real caller).
- `write_tensor_value`/`write_tensor_value_admitted` can report a genuine Provider-native failure, distinguishable from a Memory Manager admission failure, following the `write_tensor`/`stage_weight` precedent already established and tested for `HostTensor`.
- Every in-tree implementor (`ReferenceCpuExecutor`, test mocks) and both external submodule implementors (`providers/cpu`, `providers/cuda` — updated in their own repositories as a documented follow-up, not this change's diff) compile against the new trait signatures.

**Non-Goals:**
- Making `RuntimeModelExecutionEngine` externally pluggable (still `pub(crate)`). This change makes the *existing* internal wiring provider-correct; it does not open a new extension point.
- Any device-resident (`TensorValue::Opaque`) allocation, CUDA-specific memory work, or new CUDA kernels. This is purely the Core prerequisite work the audit and the `implement-cuda-provider-baseline` change both identified as blocking, not the device-residency proof itself.
- Redesigning `write_tensor_admitted` (the already-adequate `HostTensor`-typed combined admit+write method) — it isn't used by the one real caller that needs rollback discipline (`stage_weight` composes `allocate` + `write_tensor` manually instead), so it has no equivalent bug to fix here.
- Solving `TensorValue::Opaque` admission sizing (no byte length is derivable from an `Opaque` value with no data). Every call site this change touches only ever produces `TensorValue::Host` today.

## Decisions

### Decision: resolve Provider binding from `ModelInstancePlacement`, threading the instance id (or resolved binding) one level deeper

`WeightMaterializationTransaction::begin()` gains an `instance: &ModelInstanceId` parameter (its caller, `materialize_model_instance_weights`, already has it) and resolves:

```rust
let provider_binding = runtime
    .model_instance(instance)?
    .definition()
    .placement
    .provider
    .clone()
    .unwrap_or_else(|| ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME));
```

`KvUpdateTransaction::begin()` needs the same value, but its call chain (`commit_generation_step` → `promote_pending_kv_resources` → `begin()`) only carries `state: &FirstNativeExecutionKvState` today, not a model-instance id. Two options were considered:

- **Thread `&ModelInstanceId` down the same three-frame chain** (mirrors the weight path exactly). Simple, consistent, but adds a parameter to an internal function signature that already has several.
- **Resolve the binding once in `execute_qwen_graph` (which already does it correctly) and carry it forward on `FirstNativeExecutionKvState` itself**, so `commit_generation_step`/`promote_pending_kv_resources` read it from state they already hold rather than re-deriving it from a freshly re-threaded id.

Chosen: **carry the already-resolved `ProviderBinding` on `FirstNativeExecutionKvState`**, set once where `execute_qwen_graph` resolves it. This avoids re-deriving the same value twice (once for graph dispatch, once for KV commit) from two different code paths that could in principle disagree, and keeps the binding tied to the state that was actually written under it rather than to whatever the Model Instance's placement says *now* (relevant if placement could ever be observed mid-generation, though it cannot be changed once a Model Instance is created — belt-and-suspenders consistency, not a currently-reachable bug). `discard_pending_kv_state` reads the same field.

### Decision: `write_tensor_value` becomes `Result<(), ProviderExecutionError>`, default `Ok(())`

Direct mirror of `write_tensor`'s existing shape and `ProviderExecutionErrorCode::MaterializationFailed` category (already named in `write_tensor`'s own doc comment as "fitting this failure exactly" — reused verbatim, no new error variant). `ReferenceCpuExecutor::write_tensor_value`'s current implementation —

```rust
pub fn write_tensor_value(&self, id: TensorResourceId, value: TensorValue) {
    if let TensorValue::Host(tensor) = value {
        self.write_tensor(id, tensor); // discards write_tensor's own Result
    }
}
```

— becomes `Ok(self.write_tensor(id, tensor))`-shaped (CPU's inherent `write_tensor` is a genuinely infallible in-memory insert, so this is a signature fix, not a behavior change for CPU); every other implementor (test mocks, and — as a follow-up in their own repos — `providers/cpu`/`providers/cuda`) updates analogously.

### Decision: `write_tensor_value_admitted` returns a new composite `TensorValueAdmissionError` rather than overloading `MemoryError`

```rust
pub enum TensorValueAdmissionError {
    Memory(MemoryError),
    Provider(ProviderExecutionError),
}
```

Alternatives considered:
- **Keep `MemoryError` only, add a new `MemoryError` variant for "Provider write failed."** Rejected: `MemoryError` is the Memory Manager's own error type: forcing a Provider-native failure through it would make the Memory Manager's error surface lie about which subsystem actually failed, exactly the "écraser une erreur CUDA dans MemoryError" anti-pattern the audit names as one of the four bad options.
- **Split into two separate trait methods the caller composes itself** (mirroring `stage_weight`'s manual `allocate` + `write_tensor` composition exactly, deprecating the combined method). Rejected for this change: it would touch every call site's control flow more invasively than necessary to fix the error-channel gap, and the combined method's *default implementation* can already provide the identical admit-then-write-with-rollback behavior generically (next paragraph) — a Provider only needs to override it if it has a reason to admit and write differently than that default composition.

The trait's default implementation does exactly what `stage_weight` does manually, generically, using the now-fallible `write_tensor_value`:

```rust
fn write_tensor_value_admitted(
    &self,
    memory: &mut MemoryManager,
    resource_id: TensorResourceId,
    value: TensorValue,
    class: MemoryAllocationClass,
    owner: MemoryAllocationOwner,
) -> Result<(), TensorValueAdmissionError> {
    let byte_size = match &value {
        TensorValue::Host(tensor) => tensor.data.len() as u64 * size_of::<f32>() as u64,
        TensorValue::Opaque => return Err(TensorValueAdmissionError::Provider(/* unsupported */)),
    };
    let allocation = memory
        .allocate(MemoryAllocationRequest::new(class, byte_size, MemoryPlacement::ProviderOwnedOpaque(/* .. */), owner))
        .map_err(TensorValueAdmissionError::Memory)?;
    if let Err(provider_error) = self.write_tensor_value(resource_id, value) {
        let _ = memory.release(allocation.id);
        return Err(TensorValueAdmissionError::Provider(provider_error));
    }
    Ok(())
}
```

`ReferenceCpuExecutor`'s existing override stays a thin override of this same shape (it already knows its own `ProviderBinding` for `MemoryPlacement`); `execute_qwen_graph_nodes`'s three call sites change their `.map_err(...)` from unconditionally producing `MemoryAdmissionFailed` to matching on the two variants: `Memory(e) -> InferenceApiError::MemoryAdmissionFailed`, `Provider(e) -> InferenceApiError::ProviderTensorWriteFailed` (both variants already exist on `InferenceApiError`, confirmed present — no new variant needed there either).

### Decision: preserve today's Reference-CPU-only behavior exactly via `unwrap_or_else`

Every fixed call site falls back to `REFERENCE_CPU_PROVIDER_NAME` when no other binding was resolved (`placement.provider` is `None`, or the `FirstNativeExecutionKvState` field was never set by a non-default path). Since Runtime today only ever registers `ReferenceCpuProvider` in every real caller of these transactions (`build_runtime_with_model_execution_engine` and siblings), this preserves bit-for-bit identical behavior for every existing test and code path — the change is additive (a second, currently-unexercised branch becomes reachable), not a behavior change for CPU-only callers. This is the concrete reason no existing test should need its *expected outcome* changed, only, where applicable, its mock's trait-method signature.

## Risks / Trade-offs

- **[Risk]** Changing two trait methods on `ProviderExecutionApi` is a breaking change to every implementor, including two out-of-tree submodules (`providers/cpu`, `providers/cuda`) this change's own diff cannot touch. → Mitigation: both external implementations are updated in their own repositories as an immediate, explicitly-tracked follow-up once this Core change's commit is available to pin against (same two-repository update discipline `SUBMODULES.md` already documents for any Core contract change); until then, `submodule-integration` CI on the old pinned commits would fail to compile against the new trait — expected and correct, not silently ignored.
- **[Risk]** Threading a `ProviderBinding` onto `FirstNativeExecutionKvState` (rather than re-deriving it from a freshly-passed instance id) means the state struct now carries one more piece of routing information alongside its tensor data. → Mitigation: it is exactly the same kind of binding metadata `ResourceAffinity`/`ModelInstancePlacement` already carry elsewhere; no new concept is introduced, only a narrower place to read an existing one from.
- **[Trade-off]** The default `write_tensor_value_admitted` implementation fails closed for `TensorValue::Opaque` (no byte-size derivable) rather than attempting a heuristic. This is deliberate — silently guessing a size for a resource a Provider hasn't produced host-visible bytes for is exactly the kind of unaccounted-for allocation the Memory Manager's admission discipline exists to prevent. Real `Opaque` admission needs an explicit size hint, which is part of the deferred device-residency work, not this change.

## Migration Plan

1. Change `ProviderExecutionApi::write_tensor_value`/`write_tensor_value_admitted` signatures and default implementations in `provider.rs`; add `TensorValueAdmissionError`.
2. Update `ReferenceCpuExecutor` (`reference_cpu.rs`) and every in-tree test mock to the new signatures.
3. Fix `WeightMaterializationTransaction::begin()`/`KvUpdateTransaction::begin()`/`discard_pending_kv_state` to resolve a real Provider binding; thread the needed data (instance id for weights, a `FirstNativeExecutionKvState` field for KV) through their call chains.
4. Update `execute_qwen_graph_nodes`'s three `write_tensor_value_admitted` call sites to the new composite-error handling.
5. Add the new `TensorValue`-path rollback regression test; run the full existing `magnetar-runtime` suite (~1000 tests per `providers/cpu`'s README) to confirm no behavior change for existing CPU-only paths.
6. Land this change; then, as immediate follow-ups (tracked, not blocking this change's own completion): update `providers/cpu` and `providers/cuda` in their own repositories to the new trait signatures and advance this repository's two gitlink pins.
7. Rollback: revert the Core commit; no persisted/migrated data is involved (in-memory Runtime state only).

## Open Questions

- Should `FirstNativeExecutionKvState`'s new Provider-binding field be `Option<ProviderBinding>` (mirroring `ModelInstancePlacement.provider`'s optionality) or required-with-a-CPU-default at construction? Leaning `Option` for symmetry with `ModelInstancePlacement`, resolved with the same `unwrap_or_else` fallback at each read site — but this is an implementation detail to confirm while writing the code, not a proposal-blocking decision.
- `TensorValueAdmissionError`'s exact shape (a two-variant enum here) vs. a single struct with an optional field — either satisfies the requirement; the enum is chosen for now as the more idiomatic Rust shape for "exactly one of two failure kinds," open to revision during implementation review.
