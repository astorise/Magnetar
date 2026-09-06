## Context

`enable-device-resident-kernel-chaining` fixed the first-native dispatch
loop's *input*-side host round-trip (a device-resident value already held
by the resolved Provider is now passed through by resource id instead of
being downloaded and re-uploaded). It deliberately deferred the *output*
side because eliminating the physical copy there looked, at first
inspection, like it required either unifying two independently-tracked
Memory Manager admissions or breaking `ProviderExecutionApi::submit_kernel`
to let a caller choose an admission's owner.

Product/architecture review rejected the "just add an `owner` parameter to
`submit_kernel`" shortcut: Memory Manager ownership is a Runtime
responsibility, and bolting a parameter onto one trait method to fix one
Provider (CUDA) would encode that mistake into the contract every future
Provider (Metal, OpenVINO, QNN, WebGPU -- all on the roadmap) has to
implement. The explicit decision: fix this now, generically, before those
Providers exist, rather than migrate all of them later. Preference order
given: (1) Runtime pre-admits the final output resource before submission
and the Provider writes directly into it; (2) if a trait change is truly
needed, make it a generic submission/output-admission context every
Provider shares; (3) if possible, keep the old method with a
default/compatibility path rather than breaking every implementor
immediately.

Re-tracing the actual code (not just the round-trip) during this design
pass found option (1) is fully achievable **without touching
`ProviderExecutionApi`'s signature at all**:

- `KernelInvocation.outputs` (what a Provider's `submit_kernel` actually
  sees) is populated by `KernelDispatchPlan::from_selection`/
  `from_prepared_node_execution` directly from
  `KernelSelectionRequest.outputs` (`kernel_dispatch.rs`:
  `invocation.outputs = request.outputs.clone()`), which
  `dispatch_reference_cpu_operator_multi` builds itself, today choosing a
  synthesized id (`{operation_id}.out`). Nothing downstream requires that
  id to be synthesized -- passing the graph edge's own final id here
  instead is a pure caller-side change.
- `execute_invocation_with_memory_manager` (duplicated in
  `providers/cpu`, `providers/cuda`, and this crate's own
  `reference_cpu.rs`) already receives `&mut MemoryManager` and already has
  `MemoryManager::tensor_residency(&TensorResourceId) -> Option<&TensorResidency>`
  available to check "has someone already admitted this id?" before
  deciding whether to self-admit.

So the actual design is simpler than the deferred write-up in
`enable-device-resident-kernel-chaining` assumed: move *who calls
`memory.allocate`* and *which id gets used* to the caller, and make the
Provider-side admission path a conditional fallback instead of removing it.

## Goals / Non-Goals

**Goals:**
- A Kernel's output is written directly under its *final* resource
  identity (a graph edge's own id in first-native's case) instead of a
  Kernel-internal id that a caller later downloads-and-re-uploads under a
  different one.
- The caller (first-native's dispatch loop, or any future generic
  dispatch orchestrator) chooses the output's `MemoryAllocationOwner`
  (e.g. `Session(cache_id)`) at admission time -- Providers stop hardcoding
  `MemoryAllocationOwner::Provider(...)` for every output.
- Zero `ProviderExecutionApi` trait signature changes. Existing callers
  (conformance tests, direct Provider dispatch outside first-native) keep
  working unmodified via the fallback self-admission path.
- A benchmark suite gives an objective, repeatable before/after measurement
  (decode latency/token, tokens/s, H2D/D2H volume and time, kernel-vs-
  transfer split) -- not a gate on this change's correctness decision, but
  required output per the product decision.
- The fix is generic: expressed once, at the Core/Runtime level, so Metal/
  OpenVINO/QNN/WebGPU inherit it automatically rather than needing their
  own copy of this reasoning later.

**Non-Goals:**
- Multi-GPU, tensor/model sharding across Devices, NVLink/P2P: explicitly
  deferred by product decision (2026-09-06), tracked against the existing
  `multi-device-placement` capability whenever a real deployment need
  drives it. Not touched here.
- True asynchronous/overlapped execution (`execution-stream`'s full
  multi-stream model): unrelated to this change, still deferred (per
  `enable-device-resident-kernel-chaining`'s own Decision 4/Non-Goals).
- Removing the Provider-side self-admission fallback entirely: keeping it
  is a deliberate, permanent compatibility path (product preference 3), not
  a temporary shim scheduled for later deletion -- conformance tests and
  any caller with no "final identity" concept of its own (there isn't
  always one -- a standalone Kernel benchmark, for instance) still need it.

## Decisions

### Decision 1: Runtime-owned output pre-admission helper, not a trait change

New function (exact home: `magnetar-runtime/src/memory.rs`, alongside
`MemoryManager`'s other allocation helpers -- not a trait method, so it is
usable identically by first-native today and any future generic dispatch
orchestrator without any Provider implementing anything new):

```rust
pub fn admit_kernel_output(
    memory: &mut MemoryManager,
    id: TensorResourceId,
    descriptor: &TensorDescriptor,
    placement: MemoryPlacement,
    owner: MemoryAllocationOwner,
    affinity: ResourceAffinity,
) -> Result<TensorResourceDescriptor, MemoryError>
```

Allocates via the existing `MemoryAllocationRequest`/`memory.allocate`,
then records `TensorResidency` for `id` (replacing-and-releasing any
allocation `id` previously held -- the same pattern
`write_tensor_admitted`/the Decision-8 leak fix already established),
and returns the `TensorResourceDescriptor` ready to hand to
`KernelSelectionRequest::with_output`.

**Alternative considered (product's option 2)**: a generic
"submission context" object threaded through a new/changed
`ProviderExecutionApi` method, explicitly carrying admission metadata into
the Provider. Rejected as unnecessary once Decision 1's mechanism was
confirmed sufficient: the Provider never needs to *decide* placement/
ownership under this design, only to *honor* an id that already has both,
which it can already discover via the same `MemoryManager` handle
`submit_kernel` already receives. Revisit only if a future Provider
genuinely cannot honor a caller-chosen id (none identified so far -- CPU,
CUDA, and the shape of Metal/OpenVINO/QNN/WebGPU's expected storage models
all key resources by `TensorResourceId` already).

### Decision 2: `execute_invocation_with_memory_manager` checks before self-admitting

All three duplicated copies (`providers/cpu/src/lib.rs`,
`providers/cuda/src/executor.rs`, `magnetar-runtime/src/reference_cpu.rs`)
change their per-output admission loop from unconditional
`memory.allocate(...)` to:

```rust
for output in &invocation.outputs {
    let resource = &output.resource;
    if let Some(existing) = memory.tensor_residency(&resource.id) {
        // Caller (Decision 1) already admitted this identity -- honor it,
        // do not admit a second, Provider-owned allocation for the same id.
        preadmitted.push((resource.id.clone(), existing.allocation));
        continue;
    }
    // Fallback: no pre-admission found (conformance tests, direct
    // dispatch, or any caller without a "final identity" concept) --
    // self-admit exactly as before, Provider-owned.
    ...
}
```

The rest of the function (execute, roll back on failure, record residency
for genuinely newly-admitted outputs) is unchanged; genuinely
pre-admitted outputs skip the `record_tensor_residency` call too (the
caller's own residency record is already authoritative and must not be
overwritten with a Provider-chosen placement/affinity that could silently
disagree with what the caller asked for).

**Alternative considered**: have the Provider always *ask* the caller
(via some new indirection) whether an id is pre-admitted, rather than
checking `MemoryManager` directly. Rejected: `tensor_residency` is already
the Runtime's own authoritative record of what's admitted; asking a
different, redundant question through a new mechanism would just
duplicate what the Memory Manager already knows.

### Decision 3: First-native threads the edge's final id through, pre-admits before dispatch

**Implementation location refined from the original plan**: admission
cannot happen in `execute_qwen_graph_nodes` *before* calling
`dispatch_qwen_graph_node`, because the output's byte size/shape (needed
by `admit_kernel_output`) is not known until the specific operator's own
shape-inference logic runs (matmul's `[rows, cols]` from its inputs,
rmsnorm/attention keeping their primary input's shape, etc.) -- and that
logic lives inside each leaf `dispatch_qwen_*` function, not the outer
loop. Duplicating it in the caller purely to admit earlier was rejected as
needless duplication. The actual shape is as follows:

1. `dispatch_qwen_graph_node` resolves, once per node before its `match`,
   the edge's target resource id (`edge.{output_edge}`), the placement
   (new `resolved_output_placement(prepared_plan, node_id)`, mirroring
   `resolved_kernel_memory_class`'s exact lookup pattern: `ProviderOwnedOpaque`
   for Reference CPU, `Device` otherwise), and the owner
   (`MemoryAllocationOwner::Session(kv_cache_id)` -- `kv_cache_id` added as
   a new parameter). Bundled as `type OutputTarget = (TensorResourceId,
   MemoryPlacement, MemoryAllocationOwner)`.
2. Each leaf function needing this (matmul, unary, binary_same_shape,
   attention, rmsnorm, embedding's inline dispatch) takes
   `output_target: Option<OutputTarget>` and, once it has computed its own
   output descriptor, calls a new shared helper
   `resolve_output_target(ctx, operation_id, descriptor, output_target)`:
   `Some` calls `admit_kernel_output` and returns the caller-chosen id;
   `None` falls back to synthesizing `{operation_id}.out` exactly as
   before (used by RoPE's internal per-head sub-dispatches and every test
   oracle, neither of which has an edge identity of their own).
3. `dispatch_reference_cpu_operator_multi`'s own output extraction changes
   from `ctx.provider.read_tensor(...)` (forced `HostTensor` download) to
   `ctx.provider.read_tensor_value(...)`, building `NodeValue` instead of
   `HostTensor` -- an `Opaque` result becomes `NodeValue::Resident` under
   the *same* id the caller pre-admitted, with no download at all. This
   required widening `NodeValue` (previously `enable-device-resident-
   kernel-chaining`'s input-only type, renamed from `NodeInputValue`) to
   flow in both directions, and changing every `dispatch_qwen_*`/
   `dispatch_reference_cpu_operator[_multi]` return type accordingly.
4. `execute_qwen_graph_nodes` computes `needs_explicit_edge_write` once per
   node (`false` iff the returned `NodeValue` is `Resident` under exactly
   `output_resource_id`) and only performs the materialize-then-
   `write_tensor_value_admitted` sequence when true -- true for KV-history-
   concatenated edges (genuinely new data) and for "rope" nodes (no
   Kernel-level output identity to begin with, always `Host`), false for
   every other node once pre-admission is in place.

**Alternative considered**: keep the Kernel-internal id and instead make
the *edge-level* write smart enough to detect "this is the same physical
resource, skip the copy." Rejected: detecting *sameness* between two
different `TensorResourceId`s (one Kernel-internal, one edge-level)
requires exactly the id-unification this decision already does directly,
just with an extra layer of indirection to reach the same place.

### Decision 4: Benchmark suite

**Scope narrowed from the original plan, documented rather than silently
reduced.** The literal ask -- decode latency/token and tokens/s "pour un
modèle représentatif" -- requires a CUDA-bound `Runtime`/`ModelInstance`
actually driving a full first-native Qwen decode step end to end. That
does not exist yet: first-native's production dispatch loop has, to date,
only ever been exercised against Reference CPU (in this repository's own
test suite); CUDA dispatch through the *full* first-native pipeline has
never run outside this change's own `OpaqueReportingExecutor`-based unit
tests, which wrap a fake Provider, not real CUDA. Building that end-to-end
capability is separate, materially larger work (a CUDA-registered
`Runtime`/fixture wiring `ModelInstance` placement to a real `CudaProvider`)
and is not part of this change.

What is genuinely achievable and real today, and what this change ships
(`providers/cuda/benches/kernel_chaining.rs`, Criterion-based,
`cargo bench` only, never part of `cargo test`/CI): a representative
chained sequence of real `CudaKernels` calls (three matmuls, matching the
E2E fixture's own activation shape) run two ways on real hardware --
- `naive_round_trip_chain`: downloads and re-uploads the intermediate
  result between each kernel, reproducing the physical cost pattern that
  existed before `enable-device-resident-kernel-chaining`/this change;
- `resident_chain`: chains the same three kernels via device buffers
  directly, matching current, actual behavior.

Reported: real H2D/D2H crossing counts and approximate byte volume for
each variant (via `CudaKernels::upload_count`/`download_count`, added by
`enable-device-resident-kernel-chaining`), Criterion wall-clock timing for
both chains, and a `cuda_kernel_vs_transfer_split` group isolating one
resident matmul's own compute time from one upload+download round trip's
own time, so the two can be read side by side.

This also supersedes the original "run once on the commit before this
fix, once after" plan: the naive/resident functions *are* that
before/after comparison, measured in the same process on the same
hardware in the same run -- a tighter comparison than diffing across
commits (no confound from a different toolchain state, thermal
conditions, or driver session), at the cost of being a synthetic
"naive" reconstruction rather than literally checking out an old commit.
Genuinely deferred, not delivered here: decode latency/token and
tokens/s for a representative model, pending the end-to-end CUDA
generation capability described above.

This is genuinely independent of Decisions 1-3's correctness case (the
Core contract violation is fixed regardless of measured cost, per product's
own explicit statement).

**Results** (real hardware, RTX 3070 Ti Laptop GPU, one 3-matmul chain,
32x256 activations): naive round-trip chain moves 2 uploads + 3 downloads
(~160 KiB) at a median 1.37 ms; resident chain moves 0 uploads + 1
download (~32 KiB) at a median 517 µs -- **~2.65x faster, 5x less data
moved, 5x fewer H2D/D2H crossings**. The isolated split shows one resident
matmul alone costs ~125 µs while one upload+download round trip alone
costs ~205 µs: at this tensor size, a single forced round trip already
costs more than the matmul it interrupts, confirming this is a real,
non-negligible physical cost the fix removes, not measurement noise. See
tasks.md task 5.5 for the full numbers.

## Risks / Trade-offs

- [Two admission code paths (pre-admitted vs. self-admitted) permanently
  coexist in `execute_invocation_with_memory_manager`] → same accepted
  category of risk as `TensorValue::Host`/`::Opaque` and
  `NodeInputResource::Fresh`/`::Resident`; mitigated by a regression test
  exercising both paths explicitly (pre-admitted via first-native, and the
  fallback via a conformance-style direct-dispatch caller that doesn't
  pre-admit).
- [A caller could theoretically pre-admit an id with a placement/class the
  Provider's Kernel can't actually satisfy] → `validate_invocation`'s
  existing memory-class/layout/dtype checks already reject an incompatible
  `KernelResource` before the Provider ever tries to write; no new
  validation gap introduced.
- [Skipping the edge-level re-write (Decision 3) changes observable timing/
  ordering of `TensorResourceProduced` causal events] → must be verified
  against the existing per-node causal-evidence chain tests
  (`reach-architecture-freeze-1` task group 17) before merging; if event
  ordering must change, that's a deliberate, reviewed change to causal
  evidence, not an incidental one.
- [Benchmark suite adds new workspace surface (bench harness, possibly a
  new crate)] → scoped as measurement-only, no production code depends on
  it; kept out of the default `cargo test`/`cargo build --workspace` critical
  path if it turns out to need CUDA hardware to run meaningfully.

## Migration Plan

1. Land Decisions 1-2 (`admit_kernel_output` helper,
   check-before-self-admit fallback) first, in isolation -- purely additive
   behavior, verified against the full existing test suite with zero
   expected regressions (no caller uses the new helper yet, so every
   existing path takes the fallback branch unchanged).
2. Land Decision 3 (first-native threads the final id through and
   pre-admits) as a second, separable step -- this is the one that actually
   changes first-native's observable behavior (no more Kernel-internal
   id, no more forced re-write for non-KV edges).
3. Re-verify the per-node causal-evidence chain (`validate_e2e_no_shortcuts`
   and friends) after step 2, given the Risk noted above.
4. Land Decision 4 (benchmark) in parallel with 1-3; run before/after once
   step 2 is in.
5. Update `providers/cpu`/`providers/cuda` submodule pins together with the
   parent commit, matching this session's established sequencing.
6. `cuda-provider`'s spec delta (already partially amended by
   `enable-device-resident-kernel-chaining`) gets a further amendment for
   "CUDA Provider Memory Manager Integration" describing the new
   caller-pre-admits contract.

## Open Questions (resolved during implementation)

- ~~Exact mechanics of "skip the edge-level re-write"~~ -- **Resolved:
  literal no-op.** `TensorResourceProduced` is emitted unconditionally
  inside `dispatch_reference_cpu_operator_multi` itself, independent of
  whatever the outer loop does with the result afterward -- confirmed by
  reading the code, not assumed. Skipping the outer loop's redundant write
  therefore causes no observability regression; no separate confirmation
  mechanism was needed.
- ~~Whether `admit_kernel_output` belongs on `MemoryManager` itself~~ --
  **Resolved: inherent method**, matching `allocate`/`record_tensor_residency`'s
  existing shape.
- **New, genuinely open**: a real device-resident-Provider integration
  test through the full `execute_qwen_graph_nodes`/`Runtime` machinery
  (proving zero additional upload/download calls end-to-end, not just via
  the mechanism-level unit tests this change shipped with) was descoped as
  disproportionate new test scaffolding for the remaining session budget --
  see tasks.md task 3.5's note. A self-contained follow-up if stronger
  end-to-end proof is wanted later.
