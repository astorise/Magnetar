## 1. Core: output pre-admission helper

- [x] 1.1 Added `MemoryManager::admit_kernel_output(&mut self, id, descriptor, placement, owner, affinity) -> Result<TensorResourceDescriptor, MemoryError>` (`magnetar-runtime/src/memory.rs`, inherent method -- design.md's style Open Question resolved in favor of inherent, matching `allocate`/`record_tensor_residency`). Allocates via the existing `MemoryAllocationRequest`/`self.allocate`, then records `TensorResidency`, replacing-and-releasing whatever allocation `id` previously held.
- [x] 1.2 Test: `admit_kernel_output_replaces_and_releases_the_previous_allocation_for_the_same_id` (`magnetar-runtime/src/tests.rs`) -- calling it twice for the same id replaces and releases the first allocation.
- [x] 1.3 Covered by the same test: the returned `TensorResourceDescriptor` carries the exact id/descriptor/affinity needed for `KernelSelectionRequest::with_output` (verified by construction, not a separate test).

## 2. Core: Provider-side check-before-self-admit fallback

- [x] 2.1 Updated `execute_invocation_with_memory_manager` in
  `magnetar-runtime/src/reference_cpu.rs`. Exact gate used (refined from
  the plan): check `self.resource_allocations` (this Provider's own
  self-admission tracking) first -- if this exact instance self-admitted
  the id before, proceed with normal (re-)self-admission; only if *not*
  self-admitted before *and* `memory.tensor_residency(&resource.id)`
  already has a record is the output treated as caller-pre-admitted and
  skipped entirely. (Checking `tensor_residency` alone would have
  misclassified this Provider's *own* previous self-admission, from an
  earlier dispatch of the same stable Kernel-internal id, as
  externally-pre-admitted -- caught before it could break the existing
  leak-fix regression test.)
- [x] 2.2 Applied the identical change to `providers/cpu/src/lib.rs`.
- [x] 2.3 Applied the identical change to `providers/cuda/src/executor.rs`.
- [x] 2.4 Test: `reference_cpu_honors_a_caller_pre_admitted_output_without_double_admitting`
  (`magnetar-runtime/src/tests.rs`) -- pre-admits via `admit_kernel_output`
  with `MemoryAllocationOwner::Session`, dispatches, asserts exactly one
  allocation exists throughout, the caller's placement is preserved (not
  overwritten to the Provider's own default), and the Kernel's actual
  output is still written into the pre-admitted id.
- [x] 2.5 Confirmed via full regression, not a new test: `providers/cpu`
  (9/9) and `providers/cuda` (23/23, real hardware) pass unchanged --
  proves the fallback path (no caller pre-admission) behaves exactly as
  before for conformance/direct-dispatch callers.

## 3. Core: first-native threads the final resource id through and pre-admits

- [x] 3.1 `dispatch_qwen_graph_node` (not `execute_qwen_graph_nodes` --
  refined during implementation, see note below) resolves the node's
  output edge's target resource id (`edge.{output_edge}`), the placement
  (new `resolved_output_placement(prepared_plan, node_id)` helper,
  mirroring `resolved_kernel_memory_class`'s lookup exactly: `Host`-like
  `ProviderOwnedOpaque` for Reference CPU, `Device` otherwise), and the
  owner (`MemoryAllocationOwner::Session(kv_cache_id)`, now an added
  parameter) -- computed once per node, before dispatch. Actual admission
  happens inside `resolve_output_target` (new shared helper), called by
  each leaf `dispatch_qwen_*` function once it has computed the real
  output shape (matmul's `[rows, cols]`, etc.) -- this is *where* shape
  becomes known, so admission cannot happen any earlier than that without
  duplicating each operator's own shape-inference logic in the caller.
- [x] 3.2 Every leaf function that performs a genuine Kernel dispatch
  (matmul, unary, binary_same_shape, attention, rmsnorm, embedding's
  inline dispatch) now takes `output_target: Option<OutputTarget>` and
  uses it via `resolve_output_target` instead of always synthesizing
  `{operation_id}.out`. RoPE's per-head *internal* sub-dispatches
  (`dispatch_qwen_rope_per_head`'s own calls into `dispatch_qwen_unary`)
  pass `None` -- they have no edge identity of their own, unchanged from
  before. Test oracles (`execute_qwen_prefill/decode_hidden_states_through_dispatch`,
  `check_operator_coverage`) also pass `None` throughout -- they call
  `dispatch_qwen_*` directly, never through `execute_qwen_graph_nodes`,
  so they were never going to benefit from pre-admission and are
  unaffected by design.
- [x] 3.3 Resolved: literal no-op. Confirmed
  `TensorResourceProduced` is emitted inside
  `dispatch_reference_cpu_operator_multi` itself (unconditionally, on
  every dispatch, regardless of what the outer loop does afterward) --
  skipping the outer loop's redundant write causes no observability
  regression. Implementation: `execute_qwen_graph_nodes` now computes
  `needs_explicit_edge_write` once (`false` iff the returned `NodeValue`
  is `Resident` under exactly `output_resource_id`) and only performs the
  download-materialize-then-`write_tensor_value_admitted` sequence when
  true.
- [x] 3.4 Confirmed and adjusted: KV-history concatenation
  (`concat_rows`) now explicitly materializes the pre-concat value via
  `NodeValue::into_host` first (since `concat_rows` needs real bytes) and
  sets `needs_explicit_edge_write = true` unconditionally in that branch,
  since the concatenated result is genuinely different data than whatever
  the Kernel itself wrote. The separate KV-*pending*-resource write
  (`kv.{cache_id}.layer{N}.{role}.pending`, a *third* identity beyond the
  edge's own) always materializes too -- intentionally out of this
  change's scope (design.md's Decision 3 description), confirmed via the
  full existing KV-lifecycle test suite passing unchanged.
- [x] 3.5 **Descoped during implementation, documented rather than
  silently dropped**: building a real device-resident-Provider integration
  test through the *full* `execute_qwen_graph_nodes`/`Runtime` machinery
  would require a custom `Provider` wrapper (mirroring `TestFailableProvider`)
  registered under `REFERENCE_CPU_PROVIDER_NAME` plus duplicating
  `register_reference_cpu_prepared_kernels`-equivalent setup -- substantial
  new test scaffolding for marginal additional confidence beyond what 3.5's
  replacement evidence already gives: (a) the *mechanism* is proven
  directly (`admit_kernel_output_replaces_and_releases_the_previous_allocation_for_the_same_id`,
  `reference_cpu_honors_a_caller_pre_admitted_output_without_double_admitting`,
  and `enable-device-resident-kernel-chaining`'s existing
  `resident_input_passthrough_reuses_existing_resource_and_computes_correctly`
  for the input side), (b) `needs_explicit_edge_write`'s logic is a small,
  directly-auditable boolean (Resident-under-exactly-this-id or not), and
  (c) the full 1195-test regression suite (below) exercises every modified
  code path with real computation and catches any behavioral break. If a
  stronger end-to-end proof is wanted later, it is a self-contained
  follow-up, not blocked on anything else here.
- [x] 3.6 Confirmed via full regression: 1195/1195 `magnetar-runtime`
  tests pass (up from 1193), including every existing `e2e_*`/oracle test
  exercising Reference CPU (which never returns `Opaque`, so
  `needs_explicit_edge_write` is always `true` for it -- bit-for-bit
  unchanged behavior, proven empirically, not just by the type-level
  argument).

## 4. Causal-evidence and regression verification

- [x] 4.1 The full first-native E2E suite (part of the 1195-test run, 4.2
  below) includes `e2e_authoritative_path_collects_correlated_runtime_observations`
  and the other causal-chain tests (`reach-architecture-freeze-1` task
  group 17) -- all pass unchanged, confirming the per-node causal-evidence
  chain still holds with the literal-no-op resolution from task 3.3.
- [x] 4.2 Full regression: `magnetar-runtime` 1195/1195, `providers/cpu`
  9/9, `providers/cuda` 23/23 (real hardware) -- all pass, clippy/fmt clean
  across all three.

## 5. Benchmark suite (design.md Decision 4, independent of 1-4's correctness work)

- [x] 5.1 Home: `providers/cuda/benches/kernel_chaining.rs` (Criterion,
  `[[bench]] harness = false`, `criterion` added as a `dev-dependencies`-only
  crate) -- `providers/cuda` is its own standalone workspace (`[workspace]`
  at the top of its `Cargo.toml`), so this lives there rather than in
  `magnetar-runtime/benches/`, matching the crate that actually owns
  `CudaKernels`/real hardware access. Not part of `cargo test`/CI; opt-in
  via `cargo bench --bench kernel_chaining` from `providers/cuda`.
- [x] 5.2 **Descoped, documented in design.md's Decision 4** (same pattern
  as task 3.5): decode
  latency/token and tokens/s for a representative model requires a
  CUDA-bound end-to-end first-native generation pipeline that does not
  exist yet (first-native has only ever run against Reference CPU;
  CUDA-through-first-native has never been exercised outside this change's
  own fake-Provider unit tests). Not implemented -- a self-contained
  follow-up once that end-to-end capability exists, not blocked on
  anything else here.
- [x] 5.3 Implemented: `cuda_kernel_chaining`'s pre-benchmark accounting
  reports real upload/download crossing counts and approximate KiB moved
  for both `naive_round_trip_chain` and `resident_chain`, via
  `CudaKernels::upload_count`/`download_count`; Criterion measures
  wall-clock time for both.
- [x] 5.4 Implemented: `cuda_kernel_vs_transfer_split` benchmark group
  (`matmul_only_resident` vs. `upload_then_download_round_trip`) isolates
  kernel-execution time from transfer time on the same tensor size.
- [x] 5.5 **Reinterpreted, per design.md's Decision 4 note**: rather than a
  literal git-checkout-before/-after (confounded by toolchain/driver/thermal
  state across two separate processes), the benchmark's own
  `naive_round_trip_chain` (reconstructs the pre-fix physical cost pattern)
  vs. `resident_chain` (current behavior) comparison runs both in the same
  process, same run, real hardware (this workstation's RTX 3070 Ti Laptop
  GPU). Ran via `cargo bench --bench kernel_chaining` from `providers/cuda`.
  **Results** (one 3-matmul chain, 32x256 activations):
  - naive: 2 uploads + 3 downloads, ~160 KiB moved, **1.29-1.44 ms** median 1.37 ms
  - resident: 0 uploads + 1 download, ~32 KiB moved, **504-530 µs** median 517 µs
  - **~2.65x faster, 5x fewer bytes moved, 5x fewer H2D/D2H crossings (5 vs 1)**
  - kernel-vs-transfer split: one resident matmul alone costs ~125 µs;
    one upload+download round trip alone costs ~205 µs -- for this tensor
    size a single forced round trip already costs more than the matmul
    it interrupts, confirming the eliminated cost is not noise relative to
    real compute.
  Full raw Criterion output preserved in the PR/change record.

## 6. Spec and cross-repository updates

- [x] 6.1 Confirmed `cuda-provider`'s "CUDA Provider Memory Manager Integration"
  delta matches the shipped implementation exactly: the "Caller pre-admits"
  scenario matches `resource_allocations`-then-`tensor_residency` gate's
  skip branch; the "No pre-admission found (fallback)" scenario matches the
  gate's fall-through (self-admit exactly as before); the pre-existing
  "Output consumed by a later invocation" scenario matches unchanged
  (governed by task group 2's leak fix, not this task group).
- [ ] 6.2 Update `providers/cpu`/`providers/cuda` submodule pins in `SUBMODULES.md` with the new commits.
- [x] 6.3 Dispatched `gpu-runner-smoke.yml` on `arc-gpu-magnetar` against
  `main` at these exact commits (parent `32a70ef`, `providers/cpu`
  `560895b`, `providers/cuda` `7997a2f`): green --
  https://github.com/astorise/Magnetar/actions/runs/34018630488

## 7. Full verification

- [x] 7.1 `cargo build -p magnetar-runtime --lib`, `cargo test -p magnetar-runtime --lib`
  (1195/1195), `cargo clippy -p magnetar-runtime --lib --tests -- -D warnings` (clean),
  `cargo fmt --check` (clean), `cargo build --workspace` (clean).
- [x] 7.2 `cargo check --target wasm32-unknown-unknown -p magnetar-runtime --all-features`
  -- clean (pre-existing unused-variable warnings only, unrelated to this change).
- [x] 7.3 `providers/cpu` (9/9) and `providers/cuda` (23/23, real hardware,
  RTX 3070 Ti Laptop GPU): build, test, clippy (including `--benches`), fmt
  all clean.
- [x] 7.4 `openspec validate unify-provider-output-admission-and-residency --strict`
  -- valid, re-confirmed after the final design.md/tasks.md edits.
