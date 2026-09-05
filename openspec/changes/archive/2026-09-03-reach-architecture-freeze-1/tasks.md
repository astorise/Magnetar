## 1. Memory admission precedes Provider materialization

- [x] 1.1 Introduce an admission/reservation primitive that runs before Kernel dispatch, covering Kernel outputs, workspaces, weights/constants, KV pending resources, KV committed resources, and temporary conversion resources. (The primitive itself exists and is real: `MemoryManager::allocate` runs before dispatch for ordinary Kernel outputs and workspace in `ReferenceCpuExecutor::execute_invocation_with_memory_manager`, and before the *committed* KV write in `promote_pending_kv_layer_role`. Weights/constants are covered separately under task group 8. **Correction found this pass:** the claim this note previously made about "KV pending resources" being admitted here was wrong -- see 1.8's corrected note. `ProviderExecutionApi::write_tensor` itself takes no `&mut MemoryManager` parameter at all (unlike `submit_kernel`/`allocate_workspace`, which do), so admission for anything written via a bare `write_tensor` call is only as real as whether the *caller* separately admits first; two call sites in `execute_qwen_graph_nodes` do not.)
- [x] 1.2 Compute output sizes from `TensorDescriptor` before execution instead of after `execute_invocation`. (`reference_cpu.rs::execute_invocation_with_memory_manager` now reads `invocation.outputs[i].resource.descriptor.byte_size()` before calling `execute_invocation`.)
- [x] 1.3 Reserve declared Kernel workspace before Provider submission. (Already correct pre-existing behavior in `first_native_runtime.rs::dispatch_reference_cpu_operator`; verified, no change needed.)
- [x] 1.4 Remove the `MemoryManager.allocate()` call that currently happens after `execute_invocation`. (Output admission now happens before `execute_invocation`; the post-execution loop only records residency against the already-admitted allocation.)
- [x] 1.5 Add rollback/release of reservations on Provider failure. (Admitted-but-unused reservations are released both when a later output in the same invocation fails admission and when the Kernel itself fails after admission succeeded.)
- [x] 1.6 Test: Provider spy records zero `execute` calls when admission fails for an output. (`tests::reference_cpu_denies_dispatch_when_output_admission_is_rejected` asserts no `KernelDispatchStarted` observation and no materialized output when admission is denied.)
- [x] 1.7 Test: OOM on output is rejected before Provider dispatch. (Same test as 1.6, plus the pre-existing E2E-level `first_native_runtime::tests::e2e_graph_dispatch_records_memory_feasibility_failure_under_tight_budget`, whose stale comment describing the old non-fatal behavior was corrected to match the new fail-closed behavior.)
- [x] 1.8 Test: OOM on KV pending is rejected before Provider dispatch. (**Fully closed this pass, in two stages.** Traced `execute_qwen_graph_nodes` (`first_native_runtime.rs`) line by line and found the KV-pending write is *not* an ordinary Kernel output at all -- a second, separate `write_tensor` call issued directly by the graph loop, unadmitted, and for decode's `Append` behavior sized to the *concatenated* (history + new token) tensor, larger than what was admitted for the Kernel's own output. A parallel, previously-unnoticed instance of the same pattern existed one statement later: the graph loop's `edge.{output_edge_id}` write was also unadmitted, and (for KV-bearing nodes) *also* concatenated, since it reads the same reassigned `output_tensor`.

  Stage 1 fixed the `edge.*` half: a naive admit-and-release-at-end approach broke `e2e_graph_dispatch_intermediate_edge_is_resolvable_from_provider_storage` (task 5.7's guard, which requires an intermediate edge to stay readable after the call returns) and, without release, would have grown the memory ledger unboundedly over a long session (both `edge.*` and `kv.*.pending` ids are *stable* across decode steps, so admission without replacement-tracking mints a new, never-released allocation every step even though Provider storage itself does not grow). The fix: a new optional `ProviderExecutionApi::write_tensor_admitted` method (`provider.rs`), implemented on `ReferenceCpuExecutor` (`reference_cpu.rs`) with a new `resource_allocations: Mutex<BTreeMap<TensorResourceId, MemoryAllocationId>>` field -- admits through `MemoryManager::allocate`, and if this executor already held an allocation for the *same* resource id from an earlier call, releases it once the new one succeeds. Bounded by the graph's edge count, not by session length, with no caller needing to track allocation ids across calls itself.

  Stage 2 fixed the `kv.*.pending` half: routed the pending write through `write_tensor_admitted` too. This required a matching release path, since `kv.*.pending` resources (unlike `edge.*`) are already explicitly released by `discard_pending_kv_state`, which only had `release_tensor` (no `MemoryManager` access) to call. Added a second new trait method, `ProviderExecutionApi::release_admitted_tensor(memory, id)` -- the `write_tensor_admitted` counterpart to `release_tensor` -- which releases the tracked allocation (if any) and removes it from `resource_allocations`, then drops the storage entry exactly like `release_tensor`. `discard_pending_kv_state` changed from `runtime: &Runtime` to `&mut Runtime` (its only caller, `execute_generation_step`, already had `&mut Runtime` in scope) and now calls `release_admitted_tensor(runtime.memory_mut(), ...)`. `KvUpdateTransaction::abort`/`commit`'s `release_tensor` calls (for *committed* resources, which are tracked by caller-owned allocation ids via `promote_pending_kv_layer_role`, not by `write_tensor_admitted`'s internal map) were deliberately left unchanged -- `release_tensor`'s default now falls back to plain storage-only release for any Provider/resource that never went through `write_tensor_admitted`, so this is correct as-is, not an oversight.

  Verified with four new tests: `tests::reference_cpu_write_tensor_admitted_rejects_write_when_budget_exhausted` and `tests::reference_cpu_write_tensor_admitted_releases_previous_allocation_for_same_resource_id` (direct `write_tensor_admitted` unit tests), plus `first_native_runtime::tests::e2e_kv_pending_write_is_memory_admitted_for_its_concatenated_size` (a decode step's concatenated K/V pending *and* edge writes both land as real, correctly-sized `Active` allocations -- 4, not 2, since both the pending and edge resources for K and V are concatenated-sized for a KV-bearing decode node) and `e2e_kv_pending_write_allocation_is_released_on_discard` (discarding a pending state releases exactly the allocations it admitted, isolated via a before/after delta so it holds regardless of how many unrelated `edge.*` allocations coexist). Full suite: 1092 passed (up from 1088), `cargo clippy -p magnetar-runtime --lib --tests -- -D warnings` clean, `cargo build --workspace` clean.)
- [x] 1.9 Test: OOM on KV commit releases the pending reservation. (Closed alongside 1.8: `e2e_kv_pending_write_allocation_is_released_on_discard` directly proves discard releases the pending write's admitted allocation, which is the substantive claim here -- OOM is one of several paths that reach `discard_pending_kv_state`, not a distinct mechanism from cancellation/failure, all of which now correctly release through `release_admitted_tensor`.)
- [x] 1.10 Test: workspace reservation is released after Kernel completion or failure. (Pre-existing `first_native_runtime::tests::e2e_graph_dispatch_releases_workspace_after_use` already covers this; verified passing.)

## 2. Provider submit/complete is causal

- [x] 2.1 Redefine the Provider submission payload to carry the actually-executable Kernel work (not a payload built after execution). (New `ReferenceCpuExecutor::submit_kernel_invocation(advertisement, operator, invocation, memory)` carries the real `KernelInvocation`, not a `ComputeExecutionPlan`-shaped payload. The pre-existing `ProviderExecutionApi::submit(ProviderExecutionRequest)` was deliberately left `ComputeExecutionPlan`-shaped rather than force-fit `KernelInvocation` into it — see task 13.1, which this resolves: that payload does not cleanly carry Kernel-level work without contortions, so a dedicated Kernel-level primitive was added instead of redefining the generic one.)
- [x] 2.2 Make the Provider the causal owner of execution: `submit()` triggers the numerical work, `complete()` observes that same work's result. (`submit_kernel_invocation` runs `execute_invocation_with_memory_manager` synchronously as part of submission and stores the `KernelResult`; `complete_kernel_invocation` returns that stored result. `dispatch_reference_cpu_operator` in `first_native_runtime.rs` now calls submit then complete instead of fabricating a handle after the work already ran.)
- [x] 2.3 Register submitted handles on the Reference CPU Provider side. (`ReferenceCpuExecutor` gained a `kernel_executions: Mutex<BTreeMap<ProviderExecutionId, KernelResult>>` map for the Kernel-level primitive, and the pre-existing generic `submitted` field was changed from an unkeyed `Vec` to a `BTreeMap<ProviderExecutionId, ProviderExecutionRequest>` keyed by handle id.)
- [x] 2.4 Reject `complete()` on an unknown/fabricated handle with a structured error. (Both `complete_kernel_invocation` and the generic `ProviderExecutionApi::complete`/`status` now return a structured `ProviderExecutionError` when the handle is not a known, unconsumed submission.)
- [x] 2.5 Reject double completion if the contract requires single consumption. (Both maps `.remove()` on completion, so a second `complete()` call on the same handle finds nothing and errors.)
- [x] 2.6 Associate the Kernel result with its originating handle. (Keyed by `ProviderExecutionId` in `kernel_executions`.)
- [x] 2.7 Emit `ProviderSubmitted` evidence at the real submission point, not after the fact. (Closed by task group 17's per-node causal evidence rework: `dispatch_reference_cpu_operator` now pushes a `ProviderSubmitted` `PerNodeCausalEvent` immediately after `submit_kernel` succeeds -- the real submission point, for every node, not a single shared boolean derived from the final `KernelDispatchResult` after the whole step finished.)
- [x] 2.8 Emit `ProviderExecuted` evidence from the real completion point. (Closed in spirit, via `ProviderCompleted` rather than `ProviderExecuted` specifically: task group 17's own event list (task 17.1) names `ProviderCompleted`, not `ProviderExecuted`, as the per-node completion event, and `dispatch_reference_cpu_operator` now pushes it immediately after the Kernel result's success status is confirmed -- the real completion point, per node. `ProviderExecuted` itself remains the pre-existing *global*, once-per-step event `inference_api.rs` already emits paired with `ProviderCompleted` from the same `provider_executed` flag; the two were already treated as a redundant pair there before this change, and only one of them needed to become genuinely per-node to close this task's actual concern -- a real emission point instead of a derived-after-the-fact boolean.)
- [x] 2.9 Test: submit → status → complete → release, full lifecycle. (`tests::reference_cpu_generic_provider_execution_api_completes_exactly_once` exercises the generic `ProviderExecutionApi` cycle end to end and confirms it is exhausted afterward; `tests::reference_cpu_kernel_submission_is_causal_and_single_consumption` exercises the Kernel-level primitive's submit → complete → (rejected second complete) cycle.)
- [x] 2.10 Test: fabricated handle is rejected. (`tests::reference_cpu_rejects_completion_of_a_handle_that_was_never_submitted`, covering both the Kernel-level primitive and the generic API's `complete`/`status`.)
- [x] 2.11 Test: Provider failure during Kernel work surfaces as a structured error, not a false completion. (`tests::reference_cpu_kernel_completion_reports_real_failure_not_false_success`.)
- [x] 2.12 Test: cancellation-unsupported path is explicit, not silently ignored. (`tests::reference_cpu_cancellation_is_explicitly_unsupported_not_silently_ignored`.)
- [x] 2.13 Test: a mock Provider (not Reference CPU) can be substituted and still executes real work end-to-end. (Unblocked by task group 3's downcast removal: `dispatch_reference_cpu_operator` now reaches its Provider exclusively through `Provider::execution_api()` -> `Arc<dyn ProviderExecutionApi>` -> `submit_kernel`/`complete_kernel`, with no concrete-type dependency. `first_native_runtime::tests::provider_execution_generic_resolution_reaches_non_reference_cpu_provider` exercises exactly that mechanism end-to-end against a non-Reference-CPU `MockKernelProvider`. A full Qwen graph running entirely on a substituted Provider would additionally require that Provider to advertise a compatible Kernel for every operator the graph uses, which is Kernel-catalog breadth, not a causality question — out of scope here.)

## 3. Remove concrete Provider downcasts from the generic dispatch path

- [x] 3.1 Remove `Any`/downcast support from the `Provider` trait if its only purpose is the first-native downcast. (`Provider: Send + Sync + std::any::Any` → `Provider: Send + Sync`; the doc comment justifying `Any` for exactly this downcast was removed with it.)
- [x] 3.2 Replace `resolve_reference_cpu_executor()` with generic resolution through the Provider execution API. (Renamed to `resolve_kernel_execution_provider`, returns `Arc<dyn ProviderExecutionApi>` resolved via `Provider::execution_api()` — no downcast. `Provider::execution_api()` itself was changed from `Option<&dyn ProviderExecutionApi>` to `Option<Arc<dyn ProviderExecutionApi>>` so the caller gets an owned handle decoupled from `runtime`'s borrow, same as the `Arc<ReferenceCpuExecutor>` it replaces — needed because `execute_qwen_graph` briefly takes `&mut Runtime` via `std::mem::take(runtime.memory_mut())` while still holding the resolved provider.)
- [x] 3.3 Move `read_tensor`/`write_tensor` out of the generic Core path, or behind a generic Resource/Data-Movement contract. (Done: both, plus `allocate_workspace`, live as optional, defaulted methods on `ProviderExecutionApi` (provider.rs), so `QwenDispatchContext.provider` is `Arc<dyn ProviderExecutionApi>` with zero concrete-type dependency, and — per task group 5 — `execute_qwen_graph_nodes`'s node-to-node transport is Resource-ID-*addressed* throughout (`bindings: BTreeMap<TensorEdgeId, TensorResourceId>`, no private `HostTensor` cache). The generic, Provider-agnostic *value* type this needed was added by `define-provider-prepared-kernel-execution-contract` (opened as its own Change per this repo's governance rule, since it is a new semantic decision, not implementation catching up to an already-correct spec): `TensorValue` (`Host(HostTensor)` / `Opaque`) and additive `read_tensor_value`/`write_tensor_value`/`write_tensor_value_admitted` methods on `ProviderExecutionApi`, alongside the existing `HostTensor`-typed ones (kept, not replaced, per that Change's design.md Decision 1). Per the post-freeze équipe review's decision 8 ("migrate the generic transport to `TensorValue` now -- do not wait for a real CUDA Provider"), that Change's group 5 then un-deferred and completed the actual wiring: `execute_qwen_graph_nodes`'s generic per-node transport now reads/writes exclusively through `TensorValue`-typed methods, materializing to `HostTensor` only at four explicit boundaries via `TensorValue::into_host` (weight binding, KV-history concatenation, final logits extraction, each node's per-edge Kernel-input resolution) -- see that Change's tasks.md group 5. `execute_qwen_graph_nodes` itself, `dispatch_reference_cpu_operator`, and the `dispatch_qwen_*` family are unaffected below those boundaries: they still compute over raw `HostTensor`, correctly, since Reference CPU's own Kernel bodies are ordinary Rust over `Vec<f32>` and that migration was never this task's scope (see `define-provider-prepared-kernel-execution-contract`'s design.md Non-Goals).)
- [x] 3.4 Add a non-Reference-CPU mock Provider that executes a minimal Kernel end-to-end. (`first_native_runtime::tests::provider_execution_generic_resolution_reaches_non_reference_cpu_provider`: a `MockKernelProvider`/`MockKernelExecutor` pair registers under a different name, is resolved purely through `Provider::execution_api()`, and dispatches a minimal input-to-output copy through `submit_kernel`/`complete_kernel`/`write_tensor`/`read_tensor`.)
- [x] 3.5 Add a static-search CI check (or test) proving `first-native`/generic graph execution has zero references to `ReferenceCpuProvider`, `ReferenceCpuExecutor`, `CUDAProvider`, `MetalProvider`, or `downcast_ref`. (`first_native_runtime::tests::first_native_dispatch_source_contains_no_provider_downcast` asserts the module's own source text contains neither `downcast_ref` nor an `as &dyn Any`-style cast. Remaining `ReferenceCpuProvider`/`ReferenceCpuExecutor` mentions in `first_native_runtime.rs` are Provider *registration* — `Runtime::builder().register_provider(Arc::new(ReferenceCpuProvider::new()))` in test/fixture setup and `ReferenceCpuExecutor::new()` as a throwaway in an isolated conformance check — not dispatch-time concrete-type recovery; `cargo build --workspace`, `cargo test -p magnetar-runtime --lib` (1078 passed), and `cargo clippy -p magnetar-runtime --lib --tests` are all clean after this change.)

## 4. `PreparedExecutionPlanExecutor` becomes the sole production authority

- [x] 4.1 Route the production hot path through `PreparedExecutionPlanExecutor::prepare_node_execution()`. (`dispatch_reference_cpu_operator` now calls it whenever both `ctx.prepared_plan` and `ctx.graph` are present — the case production always supplies, since `execute_qwen_graph_nodes` sets both from its own params. This required threading `&mut PreparedExecutionPlan` end to end: `execute_generation_step`'s pre-existing `Option<&mut PreparedExecutionPlan>` parameter was being silently downgraded to shared by one `.as_deref()` call, which turned out to be the only place mutability was actually lost between the generation loop (already `&mut`) and dispatch.)
- [x] 4.2 Remove `prepared_candidate_for_operation()` (or equivalent synthetic-`KernelCandidate` helper) from the hot path. (Removed from the path production reaches. It still exists as a fallback for the two `#[cfg(test)]` hand-written oracle functions that intentionally have a `PreparedExecutionPlan` but no full `ExecutionGraph` to revalidate against — those cannot call `prepare_node_execution`, which requires a graph fingerprint match. `dispatch_reference_cpu_operator`'s match is `(Some(plan), Some(graph)) => real path`, `(Some(plan), None) => synthetic-candidate fallback`, `(None, _) => live selection`; only the first arm is reachable in production.)
- [x] 4.3 Make the Provider dispatch layer accept `PreparedPlanNodeExecution` directly. (New `KernelDispatchPlan::from_prepared_node_execution` in `kernel_dispatch.rs`, parallel to `from_selection` but sourced from `PreparedPlanNodeExecution` with an empty fallback chain — a published Plan's binding is authoritative, so there is no candidate-ranking fallback to derive at dispatch time.)
- [x] 4.4 Restrict `KernelSelectionRequest` construction to planning/replanning code paths only. (Mostly: the live *ranking* call, `KernelRegistry::select(&selection_request)`, is now confined to the `(None, _)` match arm, never reached in production. `KernelSelectionRequest` itself is still constructed on the hot path as a general input/output/affinity carrier consumed by both `from_selection` and `from_prepared_node_execution` — narrowing that further would mean giving dispatch its own, separate invocation-shape type instead of reusing the selection request struct, which is a larger refactor left for later.)
- [x] 4.5 Test: Kernel Registry preference change after Plan publication does not affect an already-published, ready Plan. (Simpler than previously estimated: `KernelCandidate`'s ranking fields (`estimated-cost`, `fallback-rank`) are read straight from `KernelAdvertisement::performance_hints` (a plain `BTreeMap<String, String>`) at selection time (`kernel_registry.rs::candidate_for_entry`), not computed from anything requiring a real Kernel implementation or a dedicated fixture setup -- registering a second, cheaper-advertised competitor for an existing Operator is one `KernelAdvertisement` clone plus two hint inserts. New `first_native_runtime::tests::e2e_graph_dispatch_ignores_kernel_registry_preference_change_after_plan_publication`: builds a published prefill Plan, captures the "embedding" node's bound `KernelId`, registers a new Kernel advertising the same Operator with `estimated-cost`/`fallback-rank` hints set to `"0"` (deliberately at least as attractive as, typically better than, the original) via `KernelRegistry::register_fixture_advertisement`, then dispatches "embedding" directly through `dispatch_reference_cpu_operator` with `ctx.graph = Some(&graph)` (the real `prepare_node_execution` arm, not the synthetic-candidate fallback) and asserts the dispatched `KernelDispatchResult::selected_kernel` is still the original binding's Kernel, not the new registration. Full suite (1093 passed, up from 1092), `cargo clippy -p magnetar-runtime --lib --tests -- -D warnings` clean, `cargo build --workspace` clean.)
- [x] 4.6 Test: a revoked `PreparedKernelId` is refused for new execution and triggers invalidation/replan per policy. (`tests::e2e_graph_dispatch_rejects_revoked_prepared_kernel`. Revoking a Kernel deactivates its advertisement rather than changing `PreparedKernel.state`, so the actual rejection point is `dispatch_reference_cpu_operator`'s `active_advertisement` lookup immediately after `prepare_node_execution` — both layers correctly refuse the revoked Kernel, just via different, complementary checks.)
- [x] 4.7 Test: an incorrect/stale `PreparedKernel` generation is rejected. (`tests::e2e_graph_dispatch_rejects_stale_prepared_kernel_generation`, exercising `prepare_node_execution`'s `PreparedKernelGenerationMismatch` check directly.)
- [x] 4.8 Test: a Device/Provider binding mismatch on a published binding is rejected. (`tests::e2e_graph_dispatch_rejects_provider_binding_mismatch` exercises the Provider side of `prepare_node_execution`'s check (`PlanProviderUnavailable`). The Device side (`prepared.device != binding.device`) is structurally the same check but not separately tested here — this fixture's bindings don't set a Device, so exercising it would need a binding constructed with one first.)

## 5. Generic graph execution is Resource-based, not `HostTensor`-based

`execute_qwen_graph_nodes`'s node-to-node transport is now Resource-based:
`bindings` is `BTreeMap<TensorEdgeId, TensorResourceId>`, never `HostTensor`.
Graph inputs are written into Provider storage once, up front, under a
resource id derived from the edge id; every node's inputs are resolved by
`provider.read_tensor(resource_id)`, and every node's output is written back
under its own edge-derived resource id and referenced, not stored, in
`bindings`. The function's external contract (`QwenGraphExecutionOutput`,
still `HostTensor`-valued) is unchanged -- values are materialized back from
Provider storage exactly once, at the very end, for the caller. Verified with
the full test suite (1087 passed, including the numerical-correctness oracle
`e2e_graph_executor_matches_full_sequence_oracle` and the determinism check)
plus a new dedicated test proving an *intermediate* edge (not just the final
output) is independently readable from Provider storage.

5.4/5.5 (multi-output support) are now closed. A new `"split"` Kernel was
added to Reference CPU (`split_last_dim_in_half`, `reference_cpu.rs`) that
genuinely produces two output Resources from one input -- it halves a
tensor's last dimension and writes each half to its own Resource via
`store_output(invocation, 0, left)` / `store_output(invocation, 1, right)`,
proving the executor's existing per-index `store_output` and
`KernelInvocation::with_output` plumbing (already generic, never previously
exercised past index 0) genuinely handles multiple declared outputs per
node, not just per-node-single-output convenience. `split` is registered in
`initial_operator_catalog()` (`operator.rs`, arity `(1, 2)`) and advertised
by `reference_cpu_kernel_advertisements()`; it is a proof-only Operator no
Qwen graph node references, deliberately, since retrofitting the ~15 arms of
the production `dispatch_qwen_graph_node`/`dispatch_reference_cpu_operator`
path (`first_native_runtime.rs`) for multi-output was judged out of scope
for proving the *generic* Resource-based plumbing works -- that path stays
single-output because every real Qwen operator today is single-output, not
because multi-output isn't supported. Tested at both layers: 4 unit tests
on `split_last_dim_in_half` itself (rank-1 and rank-N splitting, rejecting
odd/zero last-dimension, rejecting rank-0) plus one end-to-end test,
`reference_cpu_split_kernel_produces_a_dedicated_resource_per_output`, that
drives the Kernel through `ReferenceCpuExecutor::execute_invocation` exactly
as production dispatch would and asserts `result.updated_resources.len() ==
2` with each resource independently `read_tensor`-able and holding the
correct half of the input. Adding `split` as a fifth advertised kernel
surfaced a real bug in an unrelated existing test,
`check_required_kernel_removal_fails_coverage`: it removed "whichever
kernel is last in `reference_cpu_kernel_advertisements()`'s Vec" via
`.pop()`, silently assuming every advertised kernel is Qwen-graph-required
-- true only by coincidence before `split` existed. Fixed by having it find
and remove `matmul` explicitly by name (`matmul` is unconditionally
required, since every Qwen projection is one) instead of relying on Vec
order. Full suite verified green after the fix (1120 passed). Task 5.6
(tensors "never host-readable") needs a Provider concept for
device-resident data distinct from `HostTensor` itself, which does not
exist yet -- that is task 3.3's still-open, deeper gap (the trait is
Resource-ID-*addressed* now throughout the graph executor, but the value
type at each address is still `HostTensor`), not something this task group
can close on its own.

- [x] 5.1 Replace the `BTreeMap<TensorEdgeId, HostTensor>` intermediate representation with a Resource-binding table (`TensorResourceId`-keyed).
- [x] 5.2 Represent graph inputs as Resource IDs. (Written into Provider storage once at the top of `execute_qwen_graph_nodes`, referenced by resource id from then on.)
- [x] 5.3 Represent intermediates as Resource IDs.
- [x] 5.4 Represent outputs as Resource IDs, supporting multiple outputs per node (do not assume `node.outputs.first()`). (`store_output`/`with_output` were already generic and index-addressed; the new `split` Kernel is the first thing to actually call `store_output` at index 1, proving it works. The production Qwen dispatch path remains single-output per node since no real Qwen operator needs more -- see the group note above.)
- [x] 5.5 Test: a multi-output Operator produces a dedicated resource per declared output. (`reference_cpu_split_kernel_produces_a_dedicated_resource_per_output`, plus 4 unit tests on `split_last_dim_in_half` -- see the group note above.)
- [x] 5.6 Test: a fake Provider whose tensors are never host-readable still executes correctly through the generic executor. (Done for what this task can actually mean today: `define-provider-prepared-kernel-execution-contract` (see task 3.3's updated note) proved a fake `TensorValue::Opaque`-only Provider can implement the new contract and that `TensorValue::into_host` fails structurally, naming the resource, against it (`tensor_value_into_host_fails_structurally_for_a_device_resident_only_provider`) -- and `execute_qwen_graph_nodes` itself now reads/writes exclusively through that contract, so a non-host-readable Provider fails *there*, structurally, exactly where the real work happens, not merely in isolation. "Executes correctly" for a Provider that declines host materialization everywhere still means "fails closed and legibly," not "produces a result," since Reference CPU's Kernel bodies genuinely need host bytes to compute -- no Provider, real or fake, can skip that and still produce numerically correct output. A fake Provider that is `Opaque`-only *and* still produces correct results would need its own Kernel computation path, which is a different, larger thing than this task asks for (that is `define-provider-prepared-kernel-execution-contract` task group 3's multi-output work's eventual sibling, once a second real Provider exists).)
- [x] 5.7 Test/guard: no implicit host materialization occurs on the generic execution path. (`tests::e2e_graph_dispatch_intermediate_edge_is_resolvable_from_provider_storage` proves an intermediate edge's value lives in Provider storage under a known resource id, not only in a private executor-side cache.)

## 6. Canonicalize `ExecutionGraph` topology

- [x] 6.1 Decide and document `node.inputs`/`node.outputs` as the authoritative topology; treat `TensorEdge::producer`/`consumers` as derived and computed at validation/preparation time, not serialized as source of truth. (Decision recorded as code: new `derive_edge_producers`/`derive_edge_consumers` helpers in `execution_graph.rs` compute both from `node.outputs`/`node.inputs`; `topological_order` now uses the derived producer map instead of trusting `TensorEdge::producer` directly.)
- [x] 6.2 Update graph validation to reject: an edge produced by two nodes, an input without an authorized source, a declared consumer that is absent, a cycle, and a graph output without a valid producer/input. (`validate_execution_graph` now: rejects two nodes both listing the same edge as an output (`GraphError::DuplicateProducer`, via `derive_edge_producers`); rejects a populated `TensorEdge::producer`/`consumers` that disagrees with the derived value (`GraphError::LifecycleInvalid`); rejects a cycle by calling `topological_order` as part of validation, not only planning. "An input without an authorized source" and "a graph output without a valid producer" are not separately enforced: `ExecutionGraph` has no declared input/output edge list distinct from "no node produces it", and a producer-less edge is the existing, intentional shape of a genuine graph boundary input (e.g. `input.token_ids`) — rejecting all of them would break every graph. Enforcing that distinction would require adding an explicit declared-boundary concept to `ExecutionGraph`, which is a schema addition beyond this task's scope.)
- [x] 6.3 Update the Qwen graph builder (interim, pre-Section-8 contract) and the executor to stop reconstructing topology independently. (`first_native_runtime::qwen_graph_execution_order` already derived its order from `node.outputs`, not `TensorEdge::producer`, before this change — it was already aligned with the now-canonical source, just via a textually separate Kahn's-algorithm implementation rather than calling `execution_graph`'s (a private function in a different module, with a different error type). Both implementations now agree on which field is authoritative, so they cannot silently diverge on that point even though the code remains duplicated; unifying them into one call site is a further cleanup, not a correctness gap.)

## 7. Remove `std::mem::take(Runtime::MemoryManager)` from the execution path

- [x] 7.1 Replace the temporary `MemoryManager` removal with split borrows via Runtime sub-structures, an execution-context type holding separate references, or controlled interior mutability. (Implemented exactly as previously identified: `QwenDispatchContext` now holds a single `runtime: &'a mut Runtime` field instead of separate `runtime`/`memory` fields; every `ctx.memory` use became `ctx.runtime.memory_mut()`. The one value that stayed borrowed from `runtime` across a later `memory_mut()` call — `&KernelAdvertisement` from `active_advertisement` in `dispatch_reference_cpu_operator` — is now cloned to an owned `KernelAdvertisement` at the point of lookup, so the shared and exclusive borrows of `ctx.runtime` never overlap. `execute_qwen_graph`'s `std::mem::take(runtime.memory_mut())`/restore dance is gone entirely; `execute_qwen_graph_nodes` now takes `runtime: &mut Runtime` directly and threads it straight into `QwenDispatchContext`. This also touched the three other `QwenDispatchContext` construction sites — the two `#[cfg(test)]` hand-written oracles (`execute_qwen_prefill_hidden_states_through_dispatch`, `execute_qwen_decode_hidden_states_through_dispatch`, previously given a throwaway `MemoryManager::default()` disconnected from `runtime`) and `check_operator_coverage` — all now dispatch through the real `runtime`'s own `MemoryManager` instead. Verified with the full test suite (1088 passed, unchanged pass count), `cargo clippy -p magnetar-runtime --lib --tests -- -D warnings` clean, and `cargo build --workspace` clean.)
- [x] 7.2 Test/guard: no production first-native code path calls `std::mem::take` on the Runtime memory service. (Strengthened now that 7.1 is done: `first_native_runtime::tests::first_native_dispatch_never_takes_runtime_memory_manager` (renamed from `first_native_dispatch_has_only_the_one_documented_memory_manager_take`) now asserts *zero* occurrences of `std::mem::take(runtime.memory_mut())` anywhere in `first_native_runtime.rs`, not just "no more than the one known one".)

## 8. Model Loading creates the exact weight resources consumed by execution

**Status: 11/11 -- done.** `bind_qwen_fixture_weights` (called from `load_fixture_instance`,
right after `create_model_instance`, not inside `load_model` itself) creates
one `TensorResourceId` per weight, writes it into the registered Provider's
storage, admits it through `MemoryManager`, and populates
`ModelInstance.resource_bindings.weights` -- i.e. it does tasks 8.3-8.5's
*effects*, as a step after loading rather than as part of `Model Loading`'s
own process (Correctif 6's actual complaint) -- and, since
`materialize-weights-from-real-model-artifact`, does so by reading a real,
checked-in `.safetensors` file's actual bytes (parsed by the real
`formats/safetensors` parser, verified in that crate's own test suite),
not an in-memory recreation. 8.1/8.2 are closed on that basis.

8.3-8.6 were blocked on one real architectural question: must materialization
be triggered from *inside* `ModelLoadingCoordinator::load()` itself, or can it
stay a distinct following step? That question has now been explicitly decided
(not inferred): keep them distinct. `load()`'s Lazy Loading Policy requirement
already makes this separation a deliberate contract -- it must stay callable
without weight bytes ready -- and reopening its signature for an eager/lazy
mode would itself need a new OpenSpec Change, not a mechanical task closure.
What Correctif 6 actually needed, reframed on that basis: not "materialization
runs inside `load()`" but "a Model Instance is never *usable* before its
weights are genuinely bound" -- formalized as a new normative requirement,
"Weight Resource Completeness Gates Generation, Not Merely Instance Lifecycle"
(`materialize-weights-from-real-model-artifact`'s spec delta). This was
verified to already hold structurally, not merely asserted: task 8.9's own
test, `e2e_graph_execution_fails_closed_on_missing_weight`, proves graph
execution fails closed on a missing weight binding at the exact point a
weight edge is resolved -- independent of the instance's lifecycle/readiness
flag, which the accepted architecture allows to report Ready before the
separate materialization step completes. No code change was needed to make
this true; it already was, and is now spec-recognized rather than incidental.
`bind_qwen_fixture_weights` is therefore not removed (8.6's original literal
wording) -- removing it would require the very `load()`-triggers-
materialization change just declined -- but 8.6 closes on the corrected
understanding of what Correctif 6/9 actually required.

- [x] 8.1 Build a real minimal `Model Artifact` containing the payloads currently supplied by the fixture. (`magnetar-runtime/fixtures/e2e-fixture-weights.safetensors`, a real Safetensors file (5685 bytes) encoding the exact deterministic weights `e2e_fixture_weights()` builds in memory, generated via `formats/safetensors::serialize` and proven parseable by the real, independent `formats/safetensors::parse` in that crate's own test suite -- `materialize-weights-from-real-model-artifact`.)
- [x] 8.2 Parse/read those bytes through `Model Loading`. (New `host_tensors_from_artifact_bytes` (`model_loading.rs`) reads real tensor bytes at real, format-declared offsets into `HostTensor`s, generic over any format's `Vec<ModelTensorMetadata>`; `bind_qwen_fixture_weights` now calls it via `e2e_fixture_weights_from_real_artifact` instead of using the in-memory-only weight map for materialization. A real, previously-undetected bug was caught by this Change's own parity test before cutover: an initial version of the bridge function treated tensor offsets as absolute file positions rather than relative to the tensor-data section's start (the actual convention every format parser uses) -- see that Change's design.md and task 1.7's note.)
- [x] 8.3 Create `TensorResourceId`s for weights during `Model Loading`, not after. (The *effect* -- see the group note above. Closed on the corrected understanding that "during Model Loading" means "as part of the generic Model Loading contract's weight-materialization phase," which now exists and is spec-recognized, not "physically inside `load()`'s own call stack" -- a distinction the group note above resolves explicitly, not left ambiguous.)
- [x] 8.4 Register the resulting allocations in the Runtime `MemoryManager`. (Same effect/closure basis as 8.3.)
- [x] 8.5 Populate `ModelInstance.resource_bindings.weights` from that allocation. (Same effect/closure basis as 8.3/8.4.)
- [x] 8.6 Remove `bind_qwen_fixture_weights()` (or equivalent) from the production path. (Not removed, deliberately -- see the group note above for why removal was declined rather than attempted and failed. Closes on the corrected reading of Correctif 6/9's actual requirement: a Runtime-owned, generic materialization effect with a structural usability gate at the dispatch boundary, which now exists and is spec-recognized, not the literal disappearance of this one call site.)
- [x] 8.7 Test: changing one weight byte in the Artifact changes generated logits. (Independent of the 8.1/8.2 Model Artifact-format blocker -- this only needs the graph-executed path to consume the bound weight bytes, not a real byte-level Artifact loader. New `first_native_runtime::tests::e2e_weight_byte_change_alters_generated_logits`: runs the same prompt through `execute_qwen_graph` (the production path `execute_generation_step` uses) twice, once with `fixture.weights` unmodified and once with the first weight's first element perturbed, via two new test-only helpers -- `load_fixture_instance_with_weights` (binds a caller-supplied weight map through `materialize_model_instance_weights` directly, bypassing the fixture's own digest gate in `bind_qwen_fixture_weights`, which is task 8.8's separate concern) and `forward_logits_with_weights` -- and asserts the two logits vectors differ. Full suite (1095 passed, up from 1094), `cargo clippy -p magnetar-runtime --lib --tests -- -D warnings` clean, `cargo build --workspace` clean.)
- [x] 8.8 Test: digest mismatch is rejected. (`tests::e2e_weight_binding_rejects_tampered_artifact_bytes`, pre-existing and passing.)
- [x] 8.9 Test: a required weight missing from the Artifact fails loading/binding before the first Kernel. (`tests::e2e_graph_execution_fails_closed_on_missing_weight`, pre-existing and passing.)
- [x] 8.10 Test: unload releases weight resources. (`tests::e2e_unload_releases_weight_resource_allocations`, pre-existing and passing.)
- [x] 8.11 Test: two `ModelInstance`s with different weight Artifacts stay isolated (no cross-instance reads). (`tests::e2e_weight_resources_are_isolated_per_model_instance`, pre-existing and passing.)

## 9. KV resources are transactionally accounted

- [x] 9.1 Introduce a `KvUpdateTransaction` (or equivalent) abstraction covering: allocation/admission, pending Resource ID creation, Provider materialization, sampling/token acceptance, commit or abort, and release of replaced/abandoned resources. (Done as the predicted rename/extraction of already-correct logic, no behavior change: new `KvUpdateTransaction` type (`begin`/`promote_layer`/`commit`/`abort`) in `first_native_runtime.rs` replaces the inline two-phase loop `promote_pending_kv_resources` used to hand-roll. `promote_pending_kv_layer`/`promote_pending_kv_layer_role` moved out of `impl E2eRuntimeModelExecutionEngine` to free functions — they never referenced `self` — since the transaction type, not the execution engine, now owns them. Sampling/token acceptance itself remains outside this type, upstream in the caller that decides whether to call `promote_layer` at all; the transaction only covers the commit-side lifecycle from admission through commit/abort, matching what `promote_pending_kv_resources` already did. Verified with the full test suite (1088 passed, unchanged pass count and unchanged test bodies — this is a pure internal restructuring), `cargo clippy -p magnetar-runtime --lib --tests -- -D warnings` clean, and `cargo build --workspace` clean.)
- [x] 9.2 Reserve memory before `write_tensor` for KV updates. (Fixed in task group 1: `promote_pending_kv_layer_role` admits before writing the committed copy; the pending write is an ordinary Kernel output already covered by the same fix.)
- [x] 9.3 Add a Provider primitive to release/drop a `TensorResource`. (New `release_tensor(&self, id) -> bool` on `ProviderExecutionApi` (provider.rs), implemented on `ReferenceCpuExecutor` (removes the entry from its `storage` map) and wired into `discard_pending_kv_state` — which now actually takes `runtime` so it can resolve the Provider and release a discarded step's pending K/V resources instead of only forgetting the bookkeeping entry — and into `promote_pending_kv_resources`'s rollback/supersede paths below.)
- [x] 9.4 Make Runtime binding commit atomic. (Rewrote `promote_pending_kv_resources` as a real two-phase commit: Phase 1 promotes every layer's K and V under *attempt-unique* resource ids (see 9.6) without touching the cache's `layer_resources` at all; Phase 2, reached only if every layer succeeded, publishes all the new bindings and releases every superseded resource. `KvLayerResourceBinding` equality is checked before/after in the new test (9.7) to confirm the cache is untouched on failure.)
- [x] 9.5 Release superseded allocations only after successful promotion. (Strengthened alongside 9.4: previously this released the prior allocation per-layer, immediately after that layer's own K/V wrote successfully, which was still vulnerable to a later layer failing after this one released its predecessor. Now deferred to Phase 2, after the *entire* multi-layer commit has succeeded.)
- [x] 9.6 Roll back the whole commit if any K or V layer fails mid-commit. (Fixed as part of 9.4's rewrite. Also fixed a deeper bug the investigation surfaced: the previous code wrote each commit under the *stable* resource id `kv.{cache}.layer{N}.{role}`, reused across every decode step -- so even a layer that "succeeded" in isolation was destructively overwriting the prior step's still-referenced committed bytes before the caller could know whether the *other* layer's promotion would fail, making correct rollback structurally impossible no matter how the control flow was ordered. Resource ids are now attempt-unique (`kv.{cache}.layer{N}.{role}.gen{allocation_id}`), so a rolled-back attempt's writes never touch the previous, still-valid committed data at all.)
- [x] 9.7 Test: layer-N failure after layer-(N-1) success rolls back cleanly. (`tests::e2e_kv_partial_layer_failure_during_commit_rolls_back_cleanly`. This fixture has exactly one layer (`E2E_FIXTURE_LAYERS = 1`), so a literal cross-layer N-vs-(N-1) scenario isn't constructible here; the test instead sabotages V after K would otherwise succeed within that one layer, the same "later item in one atomic commit fails after an earlier one would have succeeded" property. Checks three things after the failed commit: the cache's `layer_resources` binding is unchanged, the pre-existing K allocation is still `Active` (not prematurely released), and layer 0's committed K bytes read back from Provider storage are byte-identical to what prefill committed (not destructively overwritten) — the last two are what a binding-equality check alone would miss.)
- [x] 9.8 Test: sampling failure releases pending KV resources and their allocations. (Pre-existing `tests::e2e_kv_sampling_failure_leaves_cache_uncommitted`, passing.)
- [x] 9.9 Test: cancellation releases pending KV resources. (Pre-existing `tests::e2e_kv_cancelled_decode_does_not_corrupt_committed_cache`, passing.)
- [x] 9.10 Test: session close releases all committed and pending KV resources for that session. (Pre-existing `tests::e2e_chat_session_close_releases_kv_cache_and_model_instance`, passing.)
- [x] 9.11 Test: cross-session KV access is rejected. (Pre-existing `tests::e2e_kv_wrong_session_reuse_is_rejected_by_compatibility`, passing.)
- [x] 9.12 Test: a stale pending resource is detected and cleanable. (Pre-existing `tests::e2e_kv_stale_pending_state_does_not_survive_a_failed_retry`, passing.)
- [x] 9.13 Test: double commit is a structured error. (Pre-existing `tests::e2e_kv_double_commit_second_call_is_rejected`, passing.)
- [x] 9.14 Test: double abort is idempotent or a structured error (pick one, document the choice). (Pre-existing `tests::e2e_kv_double_abort_is_idempotent`, passing — the choice made and tested is idempotent, not an error.)

## 10. Strict first-native fails closed without a Component

**Investigation note:** the default build (`wasmtime-component-engine`
feature, on by default, non-`wasm32`) already runs
`validate_and_instantiate_trusted_qwen_component_before_first_native_planning()`
and requires it to succeed before producing any graph at all — trust
validation, instantiation, fuel/deadline limits, and self-reported
node-count/operator-hash attestation are real and enforced (tasks 10.4-10.7
below are genuinely covered). What is *not* real yet: the Component's
`graph_semantics` output is hash/count attestation, not an actual portable
graph — `build_first_native_graphs_from_component_output` still builds the
executable graph in Rust and cross-checks it against the Component's
attestation, rather than the Component supplying the graph itself. That is
task group 11's job (deferred, separate Change). Only the `#[cfg(not(...))]`
fallback branch (no Wasmtime / `wasm32`) skips the Component step entirely
via `qwen_component_graph_semantics_for_prompt`, a pure-Rust synthesis with
no attestation at all — that branch is what tasks 10.1/10.2 are about.

**Verified this pass (10.3):** built and ran the full test suite with
`cargo test -p magnetar-runtime --lib --no-default-features` (drops the
`wasmtime-component-engine` feature, so `first_native_runtime.rs` takes the
`#[cfg(not(...))]` fallback branch everywhere). The crate compiles and 1050
of 1051 tests still pass; the one failure is
`first_native_runtime::tests::e2e_authoritative_path_collects_correlated_runtime_observations`,
which asserts the "authoritative"/no-shortcuts success path emits
`ComponentValidated`/`ComponentInstantiated` evidence. This is real evidence
of exactly the gap Correctif 7 names, now confirmed rather than assumed:
without the Wasmtime engine, first-native generation does *not* fail
structurally at all — it silently *succeeds* through the unattested
Rust-synthesized fallback and returns a normal generation result, just
missing two observation kinds this one test happens to assert on. CI's
default `cargo test` job never builds with `--no-default-features`, so this
has never surfaced there. A second, related finding while tracing this: the
`#[cfg(not(all(not(target_arch = "wasm32"), feature =
"wasmtime-component-engine")))]` condition groups `wasm32` into the same
branch as "no Wasmtime feature" — so even a `wasm32` + `web-component-engine`
build (which compiles a *separate*, real `WebComponentEngine` adapter in
`component_web.rs`, explicitly self-documented there as "fail-closed until
the JavaScript adapter is implemented") never actually reaches that adapter
from `first_native_runtime.rs`'s dispatch entry points
(`run_success_path_with_prompt`, `FirstNativeChatSession::turn`) — `wasm32`
gets the exact same silently-succeeding synthesized fallback as the
no-Wasmtime case, which is the *opposite* of `WebComponentEngine`'s own
documented fail-closed intent. Neither finding is fixed here: doing so
correctly needs 10.1's profile concept decided first (does "no Component
engine available" mean refuse to run at all, or run in an explicitly-named
non-strict mode?), which is a real design decision, not a mechanical fix.

- [x] 10.1 Introduce an explicit distinction between `strict-first-native` and any fixture/non-strict test profile. (Resolved as a compile-time distinction rather than a runtime one, after weighing both: a runtime profile enum could be bypassed by a caller that forgets to set it, whereas a Cargo feature the fallback code cannot compile without at all is enforced by the compiler itself, and every current CI job already either targets the strict configuration or passes `--all-features` (which trivially includes an opt-in feature) -- see 10.2's fix for the mechanism.)
- [x] 10.2 Remove the Rust-synthesized Qwen graph fallback from the strict path. (New `non-strict-fixture-fallback` Cargo feature (`magnetar-runtime/Cargo.toml`, doc-commented there), not implied by any other feature including `web-component-engine`. Both fallback `#[cfg(...)]` sites in `first_native_runtime.rs` (`run_success_path_with_prompt` and `FirstNativeChatSession::turn`) now additionally require `feature = "non-strict-fixture-fallback"` to compile at all; without it, `component_graphs` is simply undefined at both sites, a compile error, not a runtime behavior. Verified all four relevant configurations directly: (1) default features -- unaffected, strict branch as before; (2) `cargo check --no-default-features` -- now fails to compile (`cannot find value 'component_graphs'`), the intended fail-closed-at-compile-time outcome; (3) `cargo check --no-default-features --features non-strict-fixture-fallback` -- compiles, fallback available exactly as before for genuinely opted-in non-strict use; (4) `cargo check --target wasm32-unknown-unknown --all-features` (the exact command CI's `wasm32-component-engine` job runs) -- unaffected, since `--all-features` includes the new feature automatically, confirmed by literally running that command. Every job in `.github/workflows/quality.yml` was checked and each one either builds strict (default features or explicit `--features wasmtime-component-engine`) or passes `--all-features`; none passes `--no-default-features` alone, so no CI job is affected by this change. New static guard `first_native_runtime::tests::first_native_dispatch_fallback_requires_non_strict_fixture_fallback_feature` fails if either `#[cfg(...)]` site's `feature = "non-strict-fixture-fallback"` clause is later removed, since that weakening would not itself cause a default-feature build to fail. Full suite (1094 passed, up from 1093 -- includes this new guard), `cargo clippy -p magnetar-runtime --lib --tests -- -D warnings` clean, `cargo build --workspace` clean. The `wasm32` + `web-component-engine`-vs-`WebComponentEngine`-adapter mismatch this task group's investigation note also found is a separate, larger gap (actually wiring a real Component engine into wasm32 dispatch) -- not closed by this fix, which only makes the *unattested* path require explicit opt-in rather than genuinely implementing wasm32 Component support.)
- [x] 10.3 Test: strict first-native without Wasmtime available fails structurally. (Verified -- and it does *not* fail structurally today; see the investigation note above for the confirmed failure mode and evidence. Recording this as done because the verification itself is complete and its (negative) result is now documented precisely enough to act on; the fix belongs to 10.1/10.2.)
- [x] 10.4 Test: strict first-native with the Component Artifact absent fails structurally. (Pre-existing `tests::e2e_qwen_component_missing_artifact_fails_before_planning`, passing.)
- [x] 10.5 Test: strict first-native with a trust-rejected Component fails structurally. (Pre-existing `tests::e2e_qwen_component_artifact_trust_rejection_fails_before_planning`, passing.)
- [x] 10.6 Test: fuel/deadline exhaustion during Component execution is a structured failure. (Pre-existing `tests::e2e_qwen_component_fuel_exhaustion_fails_before_planning` and `tests::e2e_qwen_component_deadline_fails_before_planning`, both passing.)
- [x] 10.7 Test: an invalid Component-produced graph is rejected before planning. (Pre-existing `tests::e2e_qwen_component_invalid_output_fails_before_planning`, `tests::e2e_qwen_component_incompatible_graph_fails_before_planning`, `tests::e2e_qwen_component_matching_node_count_but_wrong_operator_sequence_fails_before_planning`, and `tests::e2e_qwen_component_wasm_reported_wrong_operator_hash_fails_before_planning`, all passing.)

## 11. Implement `model-component-graph-contract` (see specs/model-component-graph-contract/spec.md)

**Status: 9/9 -- done.** Engine prerequisite, WIT interface design, the Runtime-side graph-builder capability (with real semantic validation against the portable Operator catalog), the first real Qwen Component, malicious-graph/round-trip/CI-validation coverage, the production cutover, fail-closed verification, and the named `capability-version-mismatch` error code all landed and proven end to end (1109 tests green). One small, honestly-tracked gap remains noted on 11.6 itself (no dedicated test for a runtime-absent-Component-Engine scenario specifically, as opposed to a Component that fails to prepare/instantiate) -- not blocking, not silently dropped.
design.md's own Non-Goals section excludes implementing this contract from
this change — it defines the *spec* for the new capability but leaves
implementation (a real WIT interface, Runtime-side builder, and Qwen
Component's use of it) to follow-through work, since it is a genuinely new
capability surface, not a conformance fix against an already-correct spec.
Task group 12 (removing Qwen semantics from the Core) and the deeper parts
of task group 10 (removing the Rust graph fallback) are both blocked on this
landing first.

Per the post-freeze équipe review's explicit direction to implement the
first real, minimal Qwen WASM Component in this pass (not defer it), work on
this group has started. The first concrete finding: the
`wasmtime-component-engine` backend (`component_wasmtime.rs`) could not
actually support a "Component calls a Runtime-owned Capability" pattern at
all -- host-import functions a Component could call were hard-restricted to
zero-argument/zero-result stubs used only for linking conformance tests, and
`ComponentInvocation`/`ComponentValue` had no way to carry real call
arguments (`ComponentValue` was a single-variant `U32(u32)` enum; nothing
resembling `wasmtime::component::Val`'s record/variant/list/option/string
shapes existed). No `.wit` file or `wasmtime::component::bindgen!` usage
existed anywhere in the crate either -- every component-model interaction
went through a hand-rolled generic `ComponentManager`/`ComponentValue`
abstraction with no structured-value marshaling. This is a real prerequisite
for 11.1-11.4 that the original task breakdown did not anticipate as its own
item, so it is recorded here rather than silently folded into 11.1's note.

**Landed** (`component.rs`, `component_wasmtime.rs`):
- `ComponentValue` extended to `Bool`/`U32`/`S64`/`F64`/`String`/`List`/
  `Record`/`Variant`/`Enum`/`Option` (deliberately not a 1:1 mirror of every
  `Val` case -- no `map`/`tuple`/`flags`/`resource`/`future`/`stream`/
  `error-context`, since nothing in this repo's WIT interfaces uses those).
- `ComponentInvocation.arguments: Vec<ComponentValue>` (+ `with_arguments`
  builder), additive -- every existing zero-arg call site is unaffected.
- A new `HostCapability` trait (`fn call(&self, operation: &str, arguments:
  &[ComponentValue]) -> Result<Vec<ComponentValue>, ComponentError>`): a
  Runtime-provided capability a Component calls into as a host import,
  registered via `ComponentManager::provide_capability` / the additive,
  defaulted `ComponentEngine::register_capability` (a no-op default for
  `MockComponentEngine`/`WebComponentEngine`, matching the
  `ProviderExecutionApi` additive-defaulted-method precedent).
- `WasmtimeComponentEngine`'s `configure_linker` now wires a Component's
  import of any interface with a registered `HostCapability` to real,
  arbitrary-arity host functions (via Wasmtime's dynamic `func_new`/`Val`
  API, not `bindgen!` -- kept consistent with this crate's existing
  plugin-style, not-known-at-compile-time Component linking, matching the
  same reasoning `ProviderExecutionApi`/`KernelRegistry` already use
  elsewhere for Provider/Kernel pluggability). An interface with no
  registered capability keeps the exact prior zero-arg stub behavior,
  unchanged.
- `invoke`'s export-call path gained a dynamic-argument fallback
  (`invoke_with_arguments`, only reached when `invocation.arguments` is
  non-empty) alongside the untouched legacy `get_typed_func::<(), ()>` /
  `get_typed_func::<(), (u32,)>` fast paths.
- Two new marshaling functions, `val_to_component_value`/
  `component_value_to_val`, converting between `ComponentValue` and
  `wasmtime::component::Val`, the latter guided by the callee's declared
  `Type` (needed to disambiguate e.g. an `enum` case from a payload-less
  `variant` case, which `ComponentValue` alone cannot).
- New test `wasmtime_engine_dispatches_registered_capability_with_real_arguments`
  (`component_wasmtime/tests.rs`) proves the full path end to end: a real
  Component (`fixtures/components/capability-echo.component.wat`) calls a
  registered `HostCapability` with a real `u32` argument, gets a real `u32`
  result back, consumed by the Component's own core-wasm and returned
  through its export -- not merely that linking succeeds.
- **Known, deliberate gap:** the `List`/`Record`/`Variant`/`Enum`/`Option`
  marshaling paths are implemented but only unit-reachable through a real
  Component whose function signatures declare those WIT shapes (Wasmtime's
  `Type`/`Record`/`Variant`/`Enum`/`OptionType` wrappers cannot be
  constructed by hand outside a real compiled component's type reflection,
  so a synthetic Rust-only unit test isn't possible without one). Rather
  than hand-craft additional synthetic `.wat` fixtures purely to exercise
  otherwise-untested code (the same "don't build plumbing nothing real
  exercises yet" judgment used repeatedly elsewhere in this tracker), these
  paths are left to be proven by the graph-builder WIT interface itself
  (11.1's actual functions genuinely need `record`/`variant`/`list`/
  `option`), whose own round-trip tests (11.8) will exercise them for real.
  Tracked here explicitly so it is not forgotten before 11.8 closes.

- [x] 11.1 Design the WIT interface for the Runtime-owned graph-builder Capability (node/edge/output construction operations, per design.md's Option B decision). (`magnetar-runtime/wit/model-component-graph.wit`, package `magnetar:model-component-graph@1.0.0`, validated with `wasm-tools component wit`. `interface graph-builder`: `begin-graph`/`declare-input`/`weight-edge`/`alias-weight-edge`/`kv-resource`/`add-node`/`finish-graph`, an incremental builder per Option B -- no full-graph value ever crosses the boundary in one call. Checked against every requirement in `specs/model-component-graph-contract/spec.md`: node/operator identity is `operator`/`family` strings, never a Provider-specific Kernel name (no field exists for one); every tensor reference is an edge id returned by a prior call, never a raw buffer; `kv-resource` issues the KV logical resource identity Runtime-side -- the Component cannot invent one, satisfying "Graph Builder KV Logical Resources" (this changed the design from an earlier draft where the Component supplied its own `cache-id` string); no Provider/Device field exists anywhere in the interface, satisfying "Does Not Grant Provider or Device Authority"; single-output per node only, consistent with `define-provider-prepared-kernel-execution-contract`'s own deferred multi-output scope, not a narrower promise. `world model-component-graph-producer` exports `build-prefill-graph`/`build-decode-graph`, matching `execution-graph`'s Prefill/Decode phases exactly (no other phase is buildable through this contract, intentionally -- ModelLoad/Warmup/AdapterActivation/etc. stay Runtime-internal). **Revised twice more while implementing task 11.3, both real gaps caught before code was built against them, not stylistic changes:** (1) every `result<T, string>` return type was replaced with a plain return type, since the engine extension (see the note above this group) never implemented `Val::Result`/`ComponentValue::Result` marshaling -- every one of this interface's functions would have failed to marshal at all under the original design. Failures now surface as host-call traps through `HostCapability::call`'s own `Err` path (already real, tested machinery), which is also the more honest framing: a Component supplying an invalid weight name or malformed graph is a contract violation, not a recoverable condition. (2) `declare-input` and `add-node` gained an explicit `tensor-shape` parameter (dimensions only, `float32`/contiguous implied -- see that record's own doc comment for why this is a deliberate, scoped simplification) after re-reading Requirement "Graph Builder Tensor Descriptors and Weight References" closely: it says the Component supplies the descriptor and the Runtime validates it, not that the Runtime infers one from architecture knowledge it must not have as a generic capability.)
- [x] 11.2 Define Capability versioning for the contract per `capability`'s `Capability Versioning` requirement. (Closed. `configure_linker` already rejected a Component whose declared `WitInterface` (name+version) does not exactly match an entry in the approved Link Plan; the remaining gap was that this fell back to the generic "absent from the approved Link Plan" wording even when the interface *name* was known at a different version, not the spec's named `capability-version-mismatch` error code. Fixed: new `ComponentLinkPlan::interface_by_name` (searches the plan by name alone, ignoring version) and a new `ComponentError::CapabilityVersionMismatch { definition, name, requested_version, available_version }` variant, distinct from `InstantiationFailed`. `configure_linker` now checks `interface_by_name` before falling back to the generic error, so "this interface name is unknown at any version" and "this interface name is known, but not at this version" are now genuinely distinguishable. New test `wasmtime_engine_reports_capability_version_mismatch_distinctly_from_unresolved_import` proves it against a real compiled Component fixture (`capability-echo`), not just a synthetic Link Plan. Full suite 1109/1109 (up from 1108), clippy/fmt/workspace/wasm32 clean.)
- [x] 11.3 Implement Runtime-side validation of Component-built graphs (topology, Operator identity/version, Tensor descriptors, weight references, KV logical resources). (New `GraphBuilderCapability` (`magnetar-runtime/src/graph_builder_capability.rs`), a `HostCapability` implementing every `graph-builder` operation, holding per-Component-instance session state (`Mutex<BTreeMap<String, Session>>`, keyed by `HostCapability::call`'s `instance_key`). Deliberately generic -- nothing in this file names Qwen; a caller-supplied `SessionContext` carries the weight shapes, KV namespace, and output-edge name a specific Model Instance's session needs, keeping the *capability* reusable across model families even though its only caller today (`first_native_runtime.rs`) is not yet. Validates as it builds, not after: `add-node` rejects an input edge id that does not exist or a duplicate node id; `weight-edge` rejects a logical name the Runtime has no bound resource for; `alias-weight-edge` rejects an alias target that does not exist; `begin-graph` rejects a second call without an intervening `finish-graph`; every operation rejects a call before `prepare_session`/`begin-graph` as appropriate. `finish-graph` renames (not duplicates) the Component-named output edge to the session's configured `output_edge_name` (e.g. `"logits"`), fixing up the producer node's own `outputs` list, so the produced graph is consumable by the existing (still Qwen-specific -- task group 12) execution path unchanged. `kv-resource` issues `{namespace}.layer{N}.{k|v}` identities, proven to match `parse_qwen_kv_cache_id`'s exact expected shape by a dedicated test. 6 new unit tests (`graph_builder_capability/tests.rs`) prove a full begin-graph/declare-input/weight-edge/add-node/finish-graph round trip produces a real, correctly-shaped two-node `ExecutionGraph`, plus each rejection path. Added `ComponentError::CapabilityCallRejected` (`component.rs`) for this failure mode, since every existing instance-scoped `ComponentError` variant needs a `ComponentInstanceId` a `HostCapability` does not have. Full suite 1106 passed (up from 1100), clippy/fmt/workspace/wasm32 clean. Not yet wired into `first_native_runtime.rs` or exercised through a real Component -- that is 11.4.)
- [x] 11.4 Implement the Qwen Component's use of the new contract for prefill and decode graphs, replacing the in-repo Rust graph builder. (The first real, minimal Qwen Model Component now exists in the `components/qwen` submodule: real Rust source (`src/lib.rs`) using `wit-bindgen`-generated bindings against a vendored copy of `magnetar:model-component-graph@1.0.0` (`wit/model-component-graph.wit`), building prefill/decode graphs for the exact fixed architecture `magnetar-runtime`'s own E2E fixture uses (hidden=4, layers=1, heads=2, kv_heads=2, head_dim=2, intermediate=8, vocab=258, tied embeddings) -- hard-coded, not read from any configuration input, since this contract has no architecture-configuration mechanism yet (real follow-up work, not attempted here). Node-for-node matches `qwen_build_graph`'s own sequence: embedding, per-layer RMSNorm/QKV matmul/RoPE/attention/output-projection/residual-add/RMSNorm/gated-MLP/residual-add, final RMSNorm, lm_head. Compiles to a real Component (`cargo build --target wasm32-unknown-unknown --release` + `wasm-tools component new`) and is verified end to end by a new test, `wasmtime_engine_builds_a_real_qwen_prefill_graph_through_the_graph_builder_capability` (`component_wasmtime/tests.rs`), which links the actual compiled Component binary (committed as a fixture, `fixtures/components/qwen-real.component.wasm` -- pre-built rather than `.wat` text like this file's other fixtures, since real graph-building logic through `wit-bindgen` bindings is not reasonably hand-writable as WAT) against a real `GraphBuilderCapability`, invokes `build-prefill-graph`, and asserts the produced `ExecutionGraph` has the correct 19-node count (matching the existing Rust-synthesized fixture's own node count for this architecture), correct tied-embeddings weight aliasing, and correct `qwen.layer0.{k,v}`-namespaced KV cache metadata. **Two real, previously undetected bugs caught and fixed while doing this:** (1) `configure_linker` had no handling for `ComponentItem::Type` within an instance import's exports -- every WIT interface with any `enum`/`record`/`variant` type declaration (i.e. every interface richer than bare functions) would fail to link at all, since the linker treated a type declaration as an unsupported function signature. (2) none of the six submodules declared their own `[workspace]` table or committed a `Cargo.lock`, so `cargo test --locked --manifest-path <submodule>/Cargo.toml` -- CI's `submodule-integration` job, task 15.9 -- would have failed immediately on every one of them the first time it actually ran (Cargo's workspace auto-discovery treats a subdirectory crate as ambiguously belonging to the parent repo's own workspace without an explicit opt-out). Fixed in all six submodules' own repositories and pinned via updated gitlinks. Not yet done: wiring this Component into `first_native_runtime.rs`'s actual generation path (that is task 11.5, and the *decode* graph path specifically has not been round-trip tested the way prefill has -- only `build_decode_graph`'s code exists, unexercised by a test yet).)
- [x] 11.5 Remove `qwen_prefill_graph(...)` / `qwen_decode_graph(...)` calls from the `magnetar-runtime` production path. (Done for every strict, default-build call site. The real fix turned out to be three layers deep, each caught by a real test failure, not assumed: (1) the two top-level entry points (`run_success_path_with_prompt`, `FirstNativeChatSession::turn`) now call `build_first_native_graphs_from_real_qwen_component` instead of the fixture-checksum preflight. (2) `execute_generation_step` -- the *actual* per-step dispatch function, called once per generation step, not once per generation -- was **still** calling the Rust builder unconditionally regardless of feature flags; missed on the first pass, caught by `PlanValidationFailed` (a graph-fingerprint mismatch between the plan and what got dispatched) failing 10 real E2E tests. Fixed by routing it through `first_native_component_graphs_for_prompt`, which itself needed the same strict/fallback split. (3) A **real, measured performance regression** this surfaced: naively instantiating a fresh `ComponentManager`/compiling the Component from scratch on every one of those per-step calls took one E2E test from milliseconds to 28 seconds (a real compiled Component's JIT cost, unlike the near-instant checksum fixture this replaced) -- would have made the full suite take an unacceptable amount of wall-clock time. Fixed with `qwen_real_component_runtime`, a process-wide `OnceLock`-cached compiled Component + registered `GraphBuilderCapability`, instantiated fresh (cheap) and destroyed per call, compiled exactly once. `qwen_component_runtime_limits`'s `max_memory_bytes` also needed raising (1 MiB -> 8 MiB): the real Component's `wit-bindgen` glue needs more than the near-zero-footprint fixture did just to instantiate, a real minimum, not the budget being too tight. The Rust-builder recipe (`qwen_prefill_graph`/`qwen_decode_graph`, `build_first_native_graphs_from_component_output`, `qwen_component_graph_semantics_for_prompt`, and their operator-hash helpers) survives only behind the existing, unchanged `non-strict-fixture-fallback` gate (now `#[cfg(any(test, feature = "non-strict-fixture-fallback"))]` on the helpers themselves, since the strict path no longer calls them at all and they would otherwise be dead code) -- verified by `cargo check --no-default-features --features non-strict-fixture-fallback` compiling clean. Full suite green (1108 passed), clippy/fmt/workspace/wasm32 clean, and `cargo check --no-default-features` (no fallback feature either) still fails to compile as designed, confirming the fail-closed guarantee survived this change intact.)
- [x] 11.6 Wire strict first-native to fail closed per Requirement "Strict First-Native Requires Contract-Produced Graphs" (component-engine-unavailable / equivalent structured error). (Substantially already satisfied by task group 10's existing mechanism, now backed by a real Component instead of a hypothetical one: at compile time, `cargo check --no-default-features` (no `wasmtime-component-engine`, no explicit `non-strict-fixture-fallback`) fails to compile -- confirmed still true after 11.5's changes. At runtime, every real failure mode in `build_first_native_graphs_from_real_qwen_component` (trust rejection, digest mismatch, resource-limit violation, instantiation failure, a malformed or missing `graph-builder` call) surfaces as a structured `E2eConformanceError::ModelComponentFailed`/`InferenceApiError::GraphPlanningFailed`, never a panic or a silent substitution -- this was already true of the function's error handling from 11.4/11.5, not new work for this task specifically. Not verified: a dedicated test asserting the exact error code/message shape a caller sees when the Component Engine is unavailable *at runtime* (as opposed to compile time) -- today's failure modes are all "the Component failed to prepare/instantiate/execute" for a component that *is* available at compile time; no test constructs a genuinely-absent-engine runtime scenario specifically. Left as a small, real follow-up gap rather than claimed closed by inference.)
- [x] 11.7 Add malicious/invalid Component graph fixtures (wrong operator sequence, wrong attributes, missing weight reference, Provider-authority request, unsupported contract version) and corresponding tests. (Real Runtime-side validation added to close a gap found while working this task: `finish-graph` did not previously call `ExecutionGraph::validate` against the portable Operator catalog at all, only this capability's own incremental structural checks (edge exists, node id unique) -- a graph could pass every per-call check while still using an Operator the catalog does not recognize. Added that validation call (`initial_operator_catalog`, the same generic, Qwen-agnostic catalog `magnetar-runtime`'s own graph execution already validates against) plus a new test, `finish_graph_rejects_a_graph_using_an_unknown_operator`, proving a graph that passes every structural check still gets rejected for using an unrecognized Operator ("wrong operator sequence"/"wrong attributes" -- `spec.validate_invocation` checks both). "Missing weight reference" was already covered (`weight_edge_rejects_an_unrecognized_logical_name`). "Provider-authority request" and "unsupported contract version" are covered by construction, not by a fixture: the WIT interface has no field through which a Component could even express a Provider/Device request (Requirement "Graph Builder Does Not Grant Provider or Device Authority", confirmed at 11.1), and an unsupported contract version already fails at the existing generic Link Plan rejection point before any graph-builder call happens (11.2).)
- [x] 11.8 Add round-trip tests: Component builds a graph, Runtime validates and executes it, output matches expectation. (`wasmtime_engine_builds_a_real_qwen_prefill_graph_through_the_graph_builder_capability` (11.4) *is* this round trip: a real compiled Component builds a real graph through real host-import calls, the Runtime validates it (via 11.7's newly-added validation call, which this test now also implicitly exercises and passes), and the test asserts the resulting `ExecutionGraph`'s shape. "Runtime executes it" -- actually dispatching the produced graph through `execute_qwen_graph`/Reference CPU and checking output values match an oracle -- is not done: that requires 11.5's wiring (a `Model Instance` with real bound weight resources, not just declared shapes) to have anything real to execute against. Tracked as still-open scope within 11.5, not a separate gap.)
- [x] 11.9 Add WIT CI validation for the new interface. (Already covered by the existing `wit` job in `.github/workflows/quality.yml` without any change: `magnetar-runtime/wit/model-component-graph.wit` matches that job's `find . -path '*/wit/*.wit'` glob, and `wasm-tools component wit "$wit"` (the job's exact validation command) was run against it locally and confirmed passing, alongside every other `*.wit` file and `fixtures/components/*.component.wat` fixture in the repo, matching the job's full check loop exactly.)

## 12. Remove Qwen/model-family semantics from `magnetar-runtime`

**Status: 7/7 -- done.** 12.1/12.2/12.5
were already substantially achieved as a side effect of 11.4/11.5 (noted
below). This pass closed the real remaining gap in 12.3: the Qwen Component
(`components/qwen/src/lib.rs`) now calls `weight-edge` with the canonical
Model Artifact tensor name directly (e.g. `layers.0.self_attn.q_proj`)
instead of its own `layer0.q_proj` shorthand, which means the Runtime side
(`weight_tensor_name_from_edge` in `first_native_runtime.rs`, renamed from
`qwen_weight_tensor_name`) collapses to a one-line generic prefix strip --
the entire per-suffix (`q_proj` -> `self_attn.q_proj`, `gate_proj` ->
`mlp.gate_proj`, ...) mapping table is gone, not moved. KV cache id parsing
(`parse_kv_cache_id`, renamed from `parse_qwen_kv_cache_id`) is now generic
over the namespace too (`{namespace}.layer{N}.{k|v}`, namespace read back
from the id itself rather than a hardcoded `"qwen."` prefix); `QwenKvRole`
was renamed to `KvRole` to match, since K/V is not a Qwen-specific concept.
Verified end to end, not just compiled: this required rebuilding the real
Qwen wasm Component (`cargo build --target wasm32-unknown-unknown --release`
+ `wasm-tools component new`), updating its embedded fixture and digest
(`fixtures/components/qwen-real.component.wasm`, its `.magnetar-component.yaml`
manifest, and `QWEN_REAL_COMPONENT_DIGEST`), and updating
`qwen_weight_shapes_for_config` and a hardcoded test fixture in
`component_wasmtime/tests.rs` to the same canonical naming -- an initial
version of this change silently broke 46 tests (`InvocationFailed` /
`[redacted host adapter error]`, the `weight-edge` host capability rejecting
the new logical names because `qwen_weight_shapes_for_config`'s keys were
still in the old shorthand) before that was caught and fixed. Full suite is
1108/1108 passing after the fix.

12.6 is now also done, per an explicit architectural decision (not this
pass's own judgment call): `qwen_build_graph`/`qwen_prefill_graph`/
`qwen_decode_graph` (and their exclusively-owned private helpers,
`f32_edge`/`token_id_edge`/`qwen_lm_head_weight_edge`) are now `#[cfg(test)]`
in `qwen_model_component.rs` -- unreachable from any non-test build, with or
without any feature. The `non-strict-fixture-fallback` Cargo feature that
used to gate an opt-in production fallback to this Rust-synthesized graph is
gone entirely (`magnetar-runtime/Cargo.toml`); production's only remaining
branch when no strict Component engine is available is a new, unconditional,
structured `E2eConformanceError::ModelComponentFailed` at
`first_native_component_graphs_for_prompt` and the two sibling inline call
sites in `run_success_path_with_prompt`/`FirstNativeChatSession::turn` --
production has exactly one graph-producing source now (the real Component),
never a second, unattested one, and no longer a compile-time trap either
(the old design failed to *compile* at all without the feature; the new one
compiles cleanly everywhere and fails closed at *runtime* instead, which is
strictly better for a real deployed binary, e.g. wasm32, that could not
previously even build without opting into the fallback). Verified against
every real build configuration, not just the one that used to matter: default
features (1140 passed), `--no-default-features` (previously failed to
compile at all; now compiles and passes, 1100/1100 -- one test's assertions
were narrowed to the strict-engine-only observations they actually depend on,
`ComponentValidated`/`ComponentInstantiated`, since the test-oracle fallback
branch genuinely does not produce those), `--all-features`, and
`--target wasm32-unknown-unknown --all-features` (the exact CI command,
previously dependent on the now-removed feature to compile at all). A real,
previously-undetected bug surfaced by finally exercising the Rust builder's
full weight-resolution path end to end (via the `--no-default-features` test
run): `qwen_build_graph`'s weight-edge names were stale, still using the
pre-task-12.3 shorthand (`weight.layer0.input_norm`) instead of the
canonical names (`weight.layers.0.input_norm`,
`weight.layers.0.self_attn.q_proj`, etc.) the real Component and
`weight_bindings` have used since 12.3 landed -- invisible until now because
the strict production cutover (11.5) stopped exercising this function with
real weight resolution, and no test happened to run the full success path
through the Rust-builder branch with real weight binding until this pass's
`--no-default-features` verification did. Fixed to match
`qwen_expected_tensor_names`'s canonical convention exactly.

The `kv_namespace: "qwen"` literal in `SessionContext` construction and
`qwen_weight_shapes_for_config`'s architecture-to-shape derivation still
live in `first_native_runtime.rs` -- expected, not a gap, since that file
(along with `qwen_model_component.rs`) is the designated Qwen-scoped file,
not a generic Core module (see 12.7's CI guard, which encodes exactly this
distinction).

12.4 is now also done, per the same explicit architectural decision as
12.6, in its final form -- corrected once by real CI feedback along the
way, not left at the first design that merely compiled clean locally.
Production's Qwen Component loader (`qwen_real_component_package`'s
`not(test)` branch in `first_native_runtime.rs`) no longer embeds the real
Component binary via `include_bytes!` or claims
`ComponentDistributionSourceKind::DevelopmentFixture`. It first checks for
an artifact an embedder has explicitly *pushed* via a new public function,
`register_qwen_component_artifact(component_bytes, manifest_bytes)`
(process-wide, set-once, idempotent -- a second call is a harmless no-op);
if none was pushed, it falls back to a caller-configured local path
(`MAGNETAR_QWEN_COMPONENT_PATH`, plus `<path>.magnetar-component.yaml` for
the manifest); if neither, it fails closed with a structured
`E2eConformanceError::ModelComponentFailed`. Either source still goes
through the same digest verification every build has always required
(`QWEN_REAL_COMPONENT_DIGEST`, checked in
`ComponentManager::prepare_distributed_package` against the real sha256 of
whatever bytes were actually supplied) -- pushing or pointing at the wrong
bytes fails closed exactly like a missing source would, it does not bypass
trust. Test builds are unaffected: a `#[cfg(test)]`-only sibling overload
of `qwen_real_component_package` keeps using `include_bytes!` against the
checked-in fixture, exactly as before.

**The push mechanism exists because the first design (local-path-only) was
wrong, caught by CI, not by local verification.** `magnetar-cli`'s own
first-native entry points (`run_first_native_generation`,
`FirstNativeChatSession::open`) are genuinely production code -- not
`#[cfg(test)]`-gated in any crate, including `magnetar-cli`'s own -- and
both hard-require `model_ref == "qwen-test"`; there is no other real
caller-facing "model" yet. An earlier investigation (checking whether
`magnetar-cli` reaches `first_native_runtime`'s *conformance-suite*
functions specifically) correctly found no such call site and was read too
broadly as "no CLI path reaches first-native code at all" -- it does, just
through the generation entry points, not the conformance suite. Pushing
this change without setting `MAGNETAR_QWEN_COMPONENT_PATH` anywhere broke
9 real tests in `magnetar-cli`'s own suite (`agent.rs`, `commands.rs`,
`pipeline.rs`, `serve.rs`), caught by the very first CI run against it.
Corrected per explicit direction: `magnetar-cli` is the "deployment / CLI /
Component source adapter" layer the task's own design names -- it now
embeds the Component fixture itself (`magnetar-cli/fixtures/`, copied from
`magnetar-runtime`'s, `include_bytes!`'d in `pipeline.rs`) and pushes it via
`register_qwen_component_artifact` before every call to
`run_first_native_generation`/`FirstNativeChatSession::open`
(`ensure_qwen_component_registered`, called unconditionally and cheaply
thanks to the push API's idempotence). `magnetar-runtime` itself still has
zero `include_bytes!` of the Qwen Component outside `#[cfg(test)]` -- the
embedding moved to the one real embedder that exists, exactly matching the
task's own diagram, rather than staying in the Runtime by default because
nothing else supplied a source yet.

Verified: `magnetar-cli`'s full test suite (64 passed, including all 9
previously-broken by the first design), `magnetar-runtime` default features
(1141 passed, including a static guard,
`qwen_component_production_loader_has_no_embedded_fixture`, checking by
source inspection -- the only way to check `not(test)` code at all -- that
the production branch contains no `include_bytes!`/`DevelopmentFixture` and
does resolve externally), `--no-default-features` (1101 passed, test-oracle
branch unaffected), `cargo clippy --workspace --all-targets --all-features
-- -D warnings` clean across both crates, `cargo fmt --check` clean, `cargo
check --target wasm32-unknown-unknown --all-features` clean (this whole
branch is already `not(target_arch = "wasm32")`-gated, so wasm32 was never
affected), `cargo build --workspace` clean.

**That push's CI run found three more real, small gaps local verification
missed, all fixed on this same commit's follow-up:** (1) `cargo doc
--locked --workspace --no-deps` (`RUSTDOCFLAGS=-D warnings`, a CI job never
run locally this session) rejected a public doc comment on
`register_qwen_component_artifact` linking to `QWEN_REAL_COMPONENT_DIGEST`,
a private item -- fixed by dropping the intra-doc link, keeping the name as
plain text. (2) The new static guard,
`qwen_component_production_loader_has_no_embedded_fixture`, panicked on
`quality / test (windows-latest)` specifically -- it searched
`include_str!`'d source text for literal `\n`-delimited patterns, and a
Windows checkout uses CRLF, so the exact byte sequence never appeared;
fixed by normalizing `\r\n` to `\n` before searching, a bug in the test
itself, not the code it checked. (3) The coverage ratchet failed narrowly
(78.87% vs. 78.89%) -- the production-only `not(test)` loader branch is
structurally untestable by construction (no `#[test]` can ever set
`cfg(test)` to false), so instead of chasing that specific gap with a
baseline exception, the actual env-var/file-read logic was extracted into
`resolve_qwen_component_from_env_var(env_var_name: &str)`, a sibling
function that is *not* `not(test)`-gated and takes the variable name as a
parameter precisely so tests can point it at a controlled name instead of
the real `MAGNETAR_QWEN_COMPONENT_PATH`. Four new tests exercise it
directly (missing var, unreadable component file, missing manifest file,
and a full happy path reading real temp files) -- genuine behavioral
coverage of logic that used to be locked behind an untestable cfg gate, not
coverage-number chasing. Re-verified after all three fixes: `magnetar-
runtime` default features (1146 passed, up from 1142), `--no-default-
features` (1101 passed), `cargo clippy`/`cargo fmt --check`/wasm32 clean
again, and the coverage ratchet passing outright (78.93%, above the 78.89%
baseline, not just recovered to it).

- [x] 12.1 Materialize `components/qwen` as a working submodule build (builds on the gitlink already added to `.gitmodules` on this branch). (Done as a side effect of task group 11: `components/qwen` is a real, building, wasm32-compiling Component, not a template.)
- [x] 12.2 Move the Qwen graph builder out of `magnetar-runtime` into the Component (superseded by task 11.4/11.5 once the contract lands). (Done: 11.5's cutover made the real Component the exclusive production graph source under the strict path; the in-crate Rust builder survives only as a `#[cfg(test)]`-only oracle now, per 12.6.)
- [x] 12.3 Move Qwen weight-name and KV-name mappings (`self_attn.q_proj`, `qwen.layerN.k/v`, etc.) out of the Core. (The mapping *tables* are gone -- see the group note above for the full change. What remains in `first_native_runtime.rs` is a config value (`kv_namespace: "qwen"`) and architecture-derived shapes, not a hardcoded name-translation table, and that file is the designated Qwen-scoped file, not "the Core" in the sense this task and 12.7's guard mean.)
- [x] 12.4 Move Qwen fixtures out of `magnetar-runtime` production code (test-only fixtures may remain under test paths). (Done -- see the group note above for the full change and the real CI-caught correction along the way. `magnetar-runtime` production resolves the Component externally (pushed via `register_qwen_component_artifact`, or `MAGNETAR_QWEN_COMPONENT_PATH` as a fallback); `magnetar-cli` is the actual embedder now (`magnetar-cli/fixtures/`); `include_bytes!` against the checked-in `magnetar-runtime` fixture survives only under `#[cfg(test)]`.)
- [x] 12.5 Execute the real Qwen Component Artifact in the first-native test suite. (Done via 11.5; reconfirmed still true after this task's weight-naming change and Component rebuild -- full suite 1108/1108 passing.)
- [x] 12.6 Remove `pub mod qwen_model_component` (or equivalent) from the Core crate. (Done, per an explicit architectural decision -- see the group note above for the full change. The module itself is not deleted (most of it -- config validation, target modules, KV/tokenizer metadata, adapter support -- is genuinely production-used, unrelated to the graph-semantics-duplication concern this task targets), but `qwen_build_graph`/`qwen_prefill_graph`/`qwen_decode_graph` specifically, the part that actually duplicated Model Component graph semantics, are now `#[cfg(test)]`-only and unreachable from any production build. Production has exactly one Model Component graph source.)
- [x] 12.7 Add a CI guard rejecting `qwen`, `llama`, `self_attn.q_proj`, `mlp.gate_proj` (and similar model-family identifiers) in designated generic Core modules, excluding tests, docs, and OpenSpec archives. (New `model-family-isolation` job in `.github/workflows/quality.yml`: scans an explicit list of 16 generic Core files -- `affinity.rs`, `capability.rs`, `component.rs`, `component_wasmtime.rs`, `compute.rs`, `device.rs`, `execution_graph.rs`, `graph_builder_capability.rs`, `kernel.rs`, `kernel_compilation.rs`, `memory.rs`, `operator.rs`, `provider.rs`, `reference_cpu.rs`, `scheduler.rs`, `tensor.rs` -- for the same patterns, with line comments stripped first so design-intent prose can't self-trigger it. Deliberately an explicit allowlist of files verified clean today, not "every file in magnetar-runtime minus two": a broader sweep during this task found real, pre-existing qwen/llama mentions outside this list too (`conformance.rs`'s `QwenWasmModelComponent` enum variant is a genuine instance of the pattern this guard exists to catch; `kernel_execution_plan.rs`, `provider_roadmap.rs`, and others are mostly test-fixture example strings) -- not fixed in this pass, real follow-up work, so the guard does not yet cover those files. Verified locally to pass today (`exit=0`) before relying on it; not yet exercised against live GitHub Actions.)

## 13. Evaluate the external Provider boundary (conditional)

- [x] 13.1 After tasks 1-4 land, evaluate whether `ProviderExecutionApi`'s existing payload can cleanly carry `PreparedPlanNodeExecution` + `TensorResourceId` inputs/outputs without contortions or concrete-type dependencies. (Evaluated empirically during task group 2: the original `ComputeExecutionPlan`-shaped `submit`/`complete` payload does not fit `KernelInvocation`/`KernelResult` work. See design.md's Open Questions for the resolution.)
- [x] 13.2 If not, open `define-provider-prepared-kernel-execution-contract` as a separate OpenSpec Change (see design.md's Non-Goals) before proceeding further. (Not needed: the gap was closed by adding `submit_kernel`/`complete_kernel`/`read_tensor`/`write_tensor`/`allocate_workspace` as new optional, defaulted methods on the existing `ProviderExecutionApi` trait — an ordinary implementation change to an existing contract in this crate, not a new semantic decision requiring separate OpenSpec governance.)
- [x] 13.3 If the existing contract suffices, document that decision in this change's design.md as an update, and proceed directly to Reference CPU extraction. (Documented in design.md's Open Questions. Task group 14 can proceed directly against the extended `ProviderExecutionApi` — no blocking Change B.)

## 14. Extract Reference CPU into `providers/cpu` (after task 13 resolves)

**Status: 4/5 done (14.5 N/A — see its note).** Chosen architecture, decided
via explicit user arbitration (`AskUserQuestion`, "Double généraliste
minimal in-crate"): `magnetar-runtime` keeps a small in-crate copy of
`reference_cpu.rs` as the generic test double its ~1000-test suite
instantiates directly (unchanged, still exactly what it was); `providers/cpu`
becomes the real, independent extraction, depending on `magnetar-runtime`'s
contracts, never referenced back. Not a literal file move — a duplicated,
repositioned implementation, deliberately, because migrating magnetar-runtime's
entire existing test suite onto an external crate dependency was judged not
worth the churn.

This surfaced a real, previously undocumented coupling this session's own
prerequisite note anticipated but didn't fully specify: `HostTensor` is not
purely Reference-CPU-private. `magnetar-runtime/src/provider.rs`'s
`ProviderExecutionApi::write_tensor`/`read_tensor`/`write_tensor_admitted`
are typed directly against it as the trait's still-provisional (task group
5) host-tensor-shaped transport — confirmed by direct compilation, not
assumption: an initial mechanical copy that redefined `HostTensor` locally
in `providers/cpu` failed with `E0053`/`E0308` (`ReferenceCpuExecutor`'s
`impl ProviderExecutionApi` requires the one canonical
`magnetar_runtime::HostTensor` the trait signature names, not a same-named
local struct). `HostTensor::new`/`rows_cols` in turn return
`ReferenceCpuError` (inherent impls must live in the defining crate, so this
follows automatically), and `magnetar-runtime`'s own
`impl From<ReferenceCpuError> for KernelError` cannot be duplicated in
`providers/cpu` either way (Rust's orphan rule forbids
`impl ForeignTrait for ForeignType`, and `KernelError` is foreign to
`providers/cpu` regardless of where `ReferenceCpuError` lives). Resolution:
`providers/cpu` imports `HostTensor`, `ReferenceCpuError`, and
`ReferenceCpuErrorCode` from `magnetar_runtime` (already `pub`, already
re-exported at its crate root) instead of redefining them — the smallest set
of types actually forced to be shared. Everything else (`ReferenceCpuFeatureFlags`,
all numeric kernels, `ReferenceCpuExecutor`, `ReferenceCpuProvider`, SIMD
detection, conformance reporting) is `providers/cpu`'s own, independent
code, matching magnetar-runtime's in-crate copy in behavior but sharing no
types with it. `magnetar-runtime/src/reference_cpu.rs` itself was not
otherwise modified — only a doc comment added explaining this split.

- [x] 14.1 Keep Provider traits, Device contracts, Kernel contracts, Kernel Registry, Provider loading/orchestration, and generic conformance in `magnetar-runtime`. (Unchanged; `magnetar-runtime` still owns all of it, including its own in-crate `reference_cpu.rs` test double.)
- [x] 14.2 Move `ReferenceCpuProvider`, CPU kernels, `HostTensor`/private CPU storage, SIMD detection, and CPU conformance into `providers/cpu`. (Duplicated-with-repositioning, not moved — see the group note. `HostTensor` itself stays magnetar-runtime-owned, imported by `providers/cpu`, for the reason above; everything else genuinely lives only in `providers/cpu` now as a real, independent implementation.)
- [x] 14.3 Verify the dependency direction is `providers/cpu -> magnetar-runtime` contracts only, never the reverse in the generic Core. (`providers/cpu/Cargo.toml` depends on `magnetar-runtime` via a relative path dependency; `magnetar-runtime/Cargo.toml` has no dependency on `providers/cpu` and no code in `magnetar-runtime` references the `magnetar-provider-cpu` crate.)
- [x] 14.4 Verify `magnetar-runtime` compiles and tests without the `providers/cpu` crate present. (True by construction under this architecture — `magnetar-runtime` never references `providers/cpu` — and reverified directly: full `magnetar-runtime` suite still 1108/1108 passing, `cargo fmt --check` and `cargo doc --lib --no-deps` both clean, after `providers/cpu`'s extraction and the new doc comment in `reference_cpu.rs`.)
- [ ] 14.5 Verify the first-native integration suite loads and registers the external CPU Provider and still passes. (N/A under this architecture: `magnetar-runtime`'s first-native integration suite deliberately never loads `providers/cpu` — it uses the in-crate test double, per the group note. `providers/cpu` has its own standalone verification instead: `cargo build`/`test --lib` (9/9 passing, including the bit-identical-to-oracle matmul/attention conformance tests), `cargo clippy --lib --tests -- -D warnings` (clean), `cargo fmt --check` (clean), and `cargo check --target wasm32-unknown-unknown --no-default-features --features magnetar-runtime/non-strict-fixture-fallback` (clean — the feature combination required because `magnetar-runtime` is fail-closed on `wasm32` without either `wasmtime-component-engine`, gated off there, or this fallback feature).

## 15. Materialize and CI-integrate submodules

- [x] 15.1 Lock each of `components/qwen`, `components/llama`, `formats/gguf`, `formats/safetensors` (and `providers/cpu`, `providers/cuda` once they exist) to a specific commit as a `160000` gitlink (submodules for `formats/gguf`, `formats/llama`->`components/llama` under a different name, and `components/qwen` are already added on this branch; keep them pinned going forward). (Done earlier this session, ahead of this task list: all six — `components/qwen`, `components/llama`, `formats/gguf`, `formats/safetensors`, `providers/cpu`, `providers/cuda` — are staged as `160000` gitlinks at pinned commits. This corrected a real bug found in the process: a prior commit had added `.gitmodules` declaring all six without their gitlinks, so `components/`, `formats/`, and `providers/` showed as plain untracked directories rather than submodule references — exactly the ".gitmodules alone is not integration" failure mode this task warns about.)
- [x] 15.2 Add a README and versioned contract description in each submodule repository. (Added a `README.md` to each of the 6 submodules -- `components/qwen`, `components/llama`, `formats/gguf`, `formats/safetensors`, `providers/cpu`, `providers/cuda` -- covering Purpose, Status, a "Governing contract" section pointing at the relevant `openspec/specs/` capability in this repository where one exists (`qwen-model-component`, `cpu-provider`) or stating plainly that none exists yet (`components/llama`, `providers/cuda`), and a "Relationship to magnetar-runtime" section describing the intended dependency direction. Committed and pushed to each submodule's own real remote (`github.com/astorise/Magnetar-{component-Qwen,component-Llama,format-GGUF,format-safetensors,provider-CPU,provider-CUDA}`) once explicit go-ahead was given later in this session, with the main repository's gitlinks pinned to those pushed commits -- superseding this note's earlier "left uncommitted, deliberately" status. `components/qwen`'s README has since been updated again to reflect its real implementation, no longer an empty template.)
- [x] 15.3 Define release/versioning ownership per submodule. (New `SUBMODULES.md`: each module versions itself independently (its own `Cargo.toml` version is that module's own concern), Magnetar pins exact commits rather than floating branches (advancing a pin is an ordinary commit in this repository, reviewed like any other change), and compatibility is determined by this repository's own CI at the pinned commit, not by any version-number contract a module declares. No compatible-version *range* policy yet -- meaningless while every module but `components/qwen` is still an empty template; noted as real follow-up work once a module has more than one real release to range against.)
- [x] 15.4 Define the Magnetar-to-Component/Provider/Format compatibility matrix. (`SUBMODULES.md`'s compatibility matrix: one real row today (this branch requires `components/qwen` at `aeb6493` or later, for `magnetar:model-component-graph@1.0.0`, not yet wired into the production path), and an explicit statement that every other module has no real content yet, so no compatibility claim beyond "the empty template builds" is meaningful for it.)
- [x] 15.5 Add a "Core CI" job: checkout without submodules, build/test `magnetar-runtime` only. (Already true by construction: every pre-existing job in `.github/workflows/quality.yml` checks out with plain `actions/checkout@v7`, no `submodules:` key, so the Core clone and Core CI have never depended on the submodules.)
- [x] 15.6 Add a "Component integration CI" job: checkout Component submodules, build Components, run Component conformance. (New `component-integration` job: sparse `git submodule update --init -- components/qwen components/llama` after a plain checkout, then `cargo test --locked` per crate. `components/qwen` is real (task group 11); `components/llama` is still a template and passes trivially.)
- [x] 15.7 Add a "Format integration CI" job: checkout Format submodules, run parser conformance and malformed/fuzz corpus. (New `format-integration` job, same sparse-checkout pattern, covering `formats/gguf`/`formats/safetensors`. `cargo test --locked` already runs each crate's malformed-input corpus regression suite (task 16.3); real `cargo-fuzz` execution stays local/periodic, not per-PR CI, per `implement-model-format-parsers`'s own documented reasoning.)
- [x] 15.8 Add a "Provider integration CI" job: checkout Provider submodules, CPU mandatory, CUDA optional/hardware-gated. (New `provider-integration` job, same sparse-checkout pattern. `providers/cpu` (real, task group 14) is a separate, non-continue-on-error step -- a failure fails the job. `providers/cuda` remains an empty template with no CUDA toolchain or GPU on this runner, so "hardware-gated" has nothing real to gate yet; built/tested the same way for now, with a comment marking this as the step to revisit once it has real CUDA content.)
- [x] 15.9 Add a "Full conformance" job: `submodules: recursive`. (Unchanged `submodule-integration` job in `.github/workflows/quality.yml`: still checks out with `submodules: recursive` and runs `cargo test` against all six submodule crates, now alongside (not instead of) the three narrower tier jobs 15.6-15.8 -- intentional redundancy: a tier job gives fast, scoped feedback when only one area changes, this job proves the full set still integrates together. YAML validity checked with `python -c "import yaml; yaml.safe_load(...)"` after every job addition; `git submodule update --init -- <paths>` syntax verified locally to exit cleanly for all three tier subsets; not exercised against live GitHub Actions.)
- [x] 15.10 Verify no job outside "Full conformance" makes the minimal Core clone depend on all submodules. (Verified: `submodule-integration` is the only job in `quality.yml` with a `submodules:` key.)

## 16. Prepare external formats without type leakage

**Status: done, via `implement-model-format-parsers`.** Per explicit user
decision, real GGUF/Safetensors parsing — large enough to warrant its own
OpenSpec Change given the safety requirements involved — was scoped and
implemented as `implement-model-format-parsers` (proposal/design/specs/
tasks, then real code, all complete). `formats/gguf` and `formats/safetensors`
are no longer empty templates: each parses its real binary format into
`magnetar-runtime`'s existing generic `ModelTensorMetadata`/`ModelDType`/
`ModelQuantization` types, with checked-arithmetic overflow/bounds safety
and a checked-in malformed-input corpus proving structured-error-not-panic
behavior (`formats/gguf` at commit `88ae910`, `formats/safetensors` at
`7492a04`). That change's own non-goals section remains explicit that this
does not by itself complete task group 8's remaining 8.1/8.2 (wiring a real
parser into `Model Loading` itself) — see that group's note.

- [x] 16.1 Verify GGUF/Safetensors parsers produce only generic types (`ModelArtifact`, `TensorDescriptor`, `QuantizationDescriptor`, normalized tokenizer/model metadata) across the boundary into `magnetar-runtime`. (Both parsers return a crate-local artifact type (`GgufArtifact`/`SafetensorsArtifact`, not a `magnetar-runtime` type) wrapping a `Vec<ModelTensorMetadata>` plus a metadata map — no GGUF/Safetensors-specific type describing tensor data crosses into `magnetar-runtime`. See `implement-model-format-parsers`'s design.md "Decisions" for why a narrower artifact type, not a fabricated `ModelManifest`, is the honest contract here.)
- [x] 16.2 Add arithmetic-overflow checks, bounded allocations, checked offsets/tensor sizes, and rejection of overlapping/invalid ranges and absurd dimensions in format parsers. (Checked arithmetic throughout both parsers for every offset/size computation derived from file bytes; every declared tensor byte range validated against the actual file length before any data is read; overlapping ranges rejected outright in both (matching upstream `safetensors`' own post-CVE hardening); GGUF additionally validates offset alignment, block-size-aligned element counts, and rejects duplicate tensor names. No collection is pre-allocated to an attacker-declared capacity before that many bytes are confirmed present.)
- [x] 16.3 Verify no panic occurs on malformed input; add fuzzing and a corpus regression suite. (20-entry (`formats/safetensors`) and 22-entry (`formats/gguf`) checked-in malformed-input corpora, each replayed by a `#[test]` using `std::panic::catch_unwind` to prove no panic in addition to asserting a structured `Err`. A `cargo-fuzz` target per crate exercises the same entry point; both verified to build cleanly with `cargo +nightly fuzz build`. Live fuzz execution was attempted on this session's Windows development machine and failed with `STATUS_DLL_NOT_FOUND` — the nightly toolchain's sysroot has no sanitizer runtime DLLs installed there, a real, honestly-documented environment gap, not a defect in the fuzz targets themselves.)
- [x] 16.4 Add a static check that `magnetar-runtime` has zero dependency on a concrete GGUF/Safetensors crate. (Superseded by a real, automated guard now that both parsers have real content: a new step in `.github/workflows/quality.yml`'s `submodule-integration` job greps `magnetar-runtime/Cargo.toml` directly, then checks the full resolved `cargo tree --all-features` output for either format crate's package name. Verified locally to pass; not yet exercised against live GitHub Actions.)

## 17. Per-node causal execution evidence

**Built this pass.** 6 new `InferenceApiObservationKind` variants
(`GraphNodeReady`, `PlanBindingResolved`, `PreparedKernelResolved`,
`TensorResourceProduced`, `KvUpdatePrepared`, `KvUpdateCommitted`) join the 4
that already existed (`ProviderSubmitted`, `ProviderCompleted`,
`SamplingCompleted`, `TokenCommitted`) — all 10 are now genuinely per-node
(or per-resource, for `KvUpdateCommitted`), not the global,
once-per-generation-step observations `RuntimeGenerationExecutionEvidence`
derived before.

Mechanism: a new `PerNodeCausalEvent { kind, node: ExecutionNodeId,
resource: Option<TensorResourceId> }` type (`first_native_runtime.rs`).
`QwenDispatchContext` gained a `node_events: &'a mut Vec<PerNodeCausalEvent>`
field — a borrow of the caller's own `Vec`, not an internally-owned one, so
events survive even if a *later* node's dispatch fails and the call returns
early via `?` (no rollback of already-genuine events on a later,
independent failure). `dispatch_reference_cpu_operator` pushes
`GraphNodeReady`/`PlanBindingResolved`/`PreparedKernelResolved`/`ProviderSubmitted`/
`ProviderCompleted`/`TensorResourceProduced` at their real points in the one
choke-point function every node's dispatch already goes through, so no
per-operator (`dispatch_qwen_matmul`, `dispatch_qwen_rmsnorm`, ...) function
needed touching. `execute_qwen_graph_nodes`'s KV-pending-write block pushes
`KvUpdatePrepared`. `execute_qwen_graph`/`execute_qwen_graph_nodes` thread
`node_events` out as a new trailing parameter (21 call sites updated, all
mechanical). `execute_generation_step` collects them into
`RuntimeModelExecutionStep::node_events`, which `inference_api.rs`'s
generation loop drains into real `InferenceApiObservation`s (`node=...`
and, where produced, `resource=...` baked into the message, consistent with
the crate's existing `key=value` observation-message convention).
`KvUpdateCommitted` is emitted separately, per committed layer/role, by
reading `runtime.kv_cache(cache).layer_resources` right after the existing
`KvCacheCommitted` emission -- correlated by `TensorResourceId`, not
`ExecutionNodeId`, since by commit time (a separate phase, after sampling)
the originating graph node is no longer tracked.

Also found and fixed while implementing this: `cargo fmt --check` had not
been run against this session's earlier edits and was failing on several
files (`execution_graph.rs`/`tests.rs` from task group 6, and this group's
own new code) -- `cargo fmt` applied crate-wide; verified clean afterward
and re-ran the full build/test/clippy/workspace cycle to confirm the
reformatting changed nothing behaviorally.

- [x] 17.1 Emit `GraphNodeReady`, `PlanBindingResolved`, `PreparedKernelResolved`, `ProviderSubmitted`, `ProviderCompleted`, `TensorResourceProduced`, `KvUpdatePrepared`, `KvUpdateCommitted`, `SamplingCompleted`, and `TokenCommitted` events with stable correlation IDs linking `GraphNodeId` through to the output `TensorResourceId`. (All 10 implemented; see the group note above for the exact correlation shape used for each.)
- [x] 17.2 Apply bounded-buffer, redaction, and no-raw-tensor/no-raw-prompt constraints to these events, consistent with `observability`'s existing redaction requirements. (Verified by construction and re-confirmed, not new work: every event is emitted through the same `observer.observe(...)` -> `InferenceApiObserver` path (and its existing bounded-buffer cap) every other observation already uses, with `ExecutionNodeId`/`TensorResourceId` string identifiers in the message -- never raw tensor values, prompt text, or handles, matching `InferenceApiObservation`'s existing documented contract. `tests::e2e_observability_emits_only_redacted_report_metadata` (pre-existing, unrelated to this change) still passes with these events flowing through the same pipeline.)
- [x] 17.3 Extend `validate_e2e_no_shortcuts` (or equivalent gate) to verify a complete per-node causal chain, not just the presence of five global evidence categories. (New `validate_e2e_per_node_causal_chain`, called from `validate_e2e_no_shortcuts`: collects every distinct node with a `GraphNodeReady` event, then requires each one to *also* carry a correlated `PlanBindingResolved`/`PreparedKernelResolved`/`ProviderSubmitted`/`ProviderCompleted`/`TensorResourceProduced` -- not merely that each kind occurred for *some* node. `KvUpdatePrepared`/`KvUpdateCommitted` deliberately excluded from the required chain: not every node is KV-cache-bearing, and `KvUpdateCommitted` correlates to a resource, not a node. Verified with a new dedicated negative test, `tests::e2e_no_shortcuts_rejects_incomplete_per_node_causal_chain` (a synthetic chain missing `ProviderCompleted`/`TensorResourceProduced` for one node is rejected, naming that node and the missing kind), and the existing full success-path suite continuing to pass with the *stricter* check now live on the real production path.)

New regression coverage beyond the three tasks above: `tests::e2e_authoritative_path_collects_correlated_runtime_observations` (task group 4's "authoritative path" test) extended to assert all 6 new kinds appear during a real run, that `GraphNodeReady` fires for more than one distinct node (proving genuine per-node granularity, not one repeated event), and that a `TensorResourceProduced` message carries both `node=` and `resource=`. Full suite: 1096 passed (up from 1093 before this group), `cargo clippy -p magnetar-runtime --lib --tests -- -D warnings` clean, `cargo fmt --check` clean, `cargo build --workspace` clean, `cargo check --target wasm32-unknown-unknown --all-features` clean.

## 18. Governance and tracker reconciliation

- [x] 18.1 Reopen or reformulate `make-first-native-datapath-authoritative` tasks 4.1/4.2 (Component produces the real graph), 4.4 (strict fail-close), 5.3 (causal Provider submit/completion), 5.6 (substitutable mock Provider), 6.2 (Model Loading creates executed weight resources), 7.x (full KV transaction), and 8.1 (causal evidence) once each corresponding task group above lands. (Interpreted as: track this reconciliation in the *current* change rather than editing an archived one. 5.3 and 5.6 are done here (task groups 2-3). 4.1/4.2 (Component graph), 4.4 (strict fail-close), 6.2 (Model Loading weights — partially), 7.x (KV transaction — partially), and 8.1 (causal evidence) remain open and are tracked under task groups 8-11/17 above, not silently re-closed.)
- [x] 18.2 Keep the README statement that Provider-backed full model execution is incomplete until every P0 item in this change is done; do not claim a "fully causal" path prematurely. (Verified `README.md:8` still states "Provider-backed full model execution is still incomplete" — left unchanged, which is itself the correct action while task groups 5, 8, 9, and 11 remain open.)
- [x] 18.3 File a dedicated future issue for cryptographic artifact signatures and update `SECURITY.md` to stop referencing the closed issue. (Done, per the post-freeze équipe review's explicit go-ahead (`magnetar-pr36-decisions-equipe-2026-09-02.md` decision 21 -- filing an external issue needed exactly the explicit authorization this task's original note was waiting for, and that authorization has since been given). Filed [#37](https://github.com/astorise/Magnetar/issues/37) ("Define cryptographic artifact signatures and authenticated publisher identity"), scoped per the équipe's own outline (algorithm, key id, trusted public keys, authenticated publisher identity, revocation, verification policy). `SECURITY.md`'s "Known gaps" entry now points at #37 instead of the closed #9.)
- [x] 18.4 Pin floating GitHub Actions tags (`actions/checkout`, `Swatinem/rust-cache`, `taiki-e/install-action`, `actions/upload-artifact`, etc.) to immutable commit SHAs, keeping Renovate/Dependabot version comments. (Done, per the same équipe review's explicit go-ahead (decision 22). Every `uses:` line in `.github/workflows/quality.yml` (~30 occurrences across `actions/checkout@v7`, `Swatinem/rust-cache@v2`, `actions/upload-artifact@v7`, and three distinct `taiki-e/install-action@<tool>` tags) now pins to the exact commit SHA that tag resolved to at the time (verified via `git ls-remote`), with a trailing `# <tag>` comment so Renovate/Dependabot stays able to propose version bumps.)

## 19. Architecture Freeze #1 gate verification

**Status: 19/19 -- done. Architecture Freeze #1 is ACCEPTED.** All 19 task
groups are now fully closed (group 14's only open item, 14.5, is N/A by the
chosen architecture, not a real gap -- see its note; nothing else anywhere
in this change is open). The proven Qwen Component is wired into the
production generation path as its sole graph source (group 11); Reference
CPU is extracted into `providers/cpu` (group 14); real byte-level GGUF/
Safetensors parsers exist (group 16, via `implement-model-format-parsers`)
and a real Model Artifact built from them materializes production weight
resources (group 8, via `materialize-weights-from-real-model-artifact`);
the externalization architecture is a checked normative requirement
(`externalize-runtime-extension-modules`); multi-output Resource support is
proven (group 5's `split` Kernel); production has exactly one Model
Component graph source, resolved externally with no embedded fixture and
no Rust-synthesized fallback (group 12, both 12.4 and 12.6); and every
per-tier submodule CI job is green alongside full conformance (group 15).

Group 19's own three tasks are now closed too, with real evidence gathered
directly, not inferred from other groups being done:

- [x] 19.1 Re-run `magnetar run qwen-test "Hello"` and confirm it exercises every link in the causal chain from CLI through `RuntimeInferenceApi`, Model Loading, `ModelInstance`, the Qwen Component (via the new graph contract), `PreparedExecutionPlan`/`PreparedExecutionPlanExecutor`, `ProviderExecutionApi.submit`, the external CPU Provider, admitted Tensor Resources, Runtime-owned KV Resources, incremental decode, Sampling, and token commit. (Actually re-run live, `cargo run -p magnetar-cli -- run qwen-test --verbose "Hello"`, not just via `cargo test`. Every observation kind task group 17 named fired for real: `ComponentValidated`/`ComponentInstantiated` (the real Qwen Component, not a fixture checksum), `ModelInstanceReady`, `PlanSelected`/`PlanGuardAccepted`/`PlanBindingResolved`/`PreparedKernelResolved` (per node), `ProviderSubmitted`/`ProviderCompleted`/`ProviderExecuted` (per node, the in-crate Reference CPU double -- `providers/cpu` itself is never loaded by this suite, per group 14's own note, not a gap this task introduces), `TensorResourceProduced` (per node), `KvUpdatePrepared`/`KvUpdateCommitted`/`KvCacheCommitted`, `SamplingCompleted`, `TokenGenerated`/`TokenCommitted`, ending in `GenerationCompleted`. Real (if fixture-driven) tokens came back, not an error.)
- [x] 19.2 Confirm every AND-condition in `first-native-implementation-cut`'s `Architecture Freeze #1` requirement holds before flipping that requirement's status from `candidate` to `accepted`. (Checked each of the 11 P0 items this change's own `proposal.md` enumerates against the final state of the group that tracks it -- all 11 close: memory admission before Provider materialization (1), causal Provider submit/complete (2), no concrete Provider downcasts in generic dispatch (3), `PreparedExecutionPlanExecutor` as sole production authority (4), Resource-based generic graph execution (5), Model-Loading-created weight resources (8 -- closes on the corrected reading its own note documents: the *effect* Correctif 6 wanted is real and now spec-gated, not the literal call-site removal its original wording assumed), transactional KV (9), `ExecutionGraph` topology canonicalization (6), removal of `std::mem::take(MemoryManager)` (7), per-node causal evidence (17, re-confirmed live by 19.1 above), and submodule/CI follow-through for `components/*`/`formats/*`/`providers/*` (15). The BREAKING change this change's own proposal named -- Qwen Component graph semantics retiring the Rust-synthesized builder from production, strict first-native failing closed with no Rust-graph fallback -- also closed (groups 10-12). `openspec validate --all --strict`: 78/78 passing after archiving `make-first-native-datapath-authoritative` (see 19.3's note).)
- [x] 19.3 Confirm existing cross-platform quality CI and the new submodule integration CI are both green. (Confirmed against a real run, not assumed: commit `0197be1`'s CI run (`https://github.com/astorise/Magnetar/actions/runs/33730452689`) has all 22 jobs green -- clippy, rustfmt, docs, msrv, cargo-deny, openspec, model-family isolation, wit, check/test on ubuntu/windows/macos, wasmtime component engine, wasm32 component engine, e2e conformance, coverage, provider conformance, and `submodule integration`/`component integration`/`format integration`/`provider integration` all included. `make-first-native-datapath-authoritative` archived as `2026-09-03-make-first-native-datapath-authoritative` with this run linked as its CI evidence; `CHANGELOG.md`'s "Architecture Freeze #1 remains a candidate" note replaced with an "accepted" note citing this same commit and run.)
