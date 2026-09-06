## Context

`docs/audits/cuda-provider-audit-synthese-2026-09-06.md` approved the CUDA
Provider standalone (`Magnetar-provider-CUDA@8eb0160`) but found first-native
still forces Host round-trips in three places once a real Device-resident
Provider is selected. This design's first draft was reviewed by product
before implementation started
(`docs/audits/revue-openspec-cuda-pre-implementation-2026-09-06.md`, verdict
**NO-GO**) and found real, specific defects, each re-verified directly
against the code below before writing this revision.

**RoPE's head_count, verified**: `first_native_runtime.rs` already contains
`qwen_rope_head_count(node, architecture)`, which branches on
`node.id.as_str().ends_with("rope_q")`/`"rope_k"` to return
`architecture.attention_head_count`/`.kv_head_count` respectively -- so
first-native's *existing* single-head-per-call dispatch already gets GQA
right, today, by id-suffix convention. This change's first draft's design
text only mentioned `attention_head_count` for the *new* single-dispatch
call, which would have silently broken K's head count for any
`kv_head_count != attention_head_count` model the moment
`dispatch_qwen_rope_per_head` was deleted -- exactly the regression the
review caught. The review's requested fix (an explicit `head_count`
graph attribute, set by the graph builder, not inferred from an id-naming
convention) is a strictly better fix than patching the id-suffix heuristic
to keep working: `qwen_model_component.rs` (`magnetar-runtime`, the actual
first-native graph source -- confirmed by reading it directly, not assumed)
builds `rope_q`/`rope_k` nodes with `a.attention_head_count`/
`a.kv_head_count` already in scope two nodes later (for `"attention"`'s own
attributes), so adding `head_count` there is a small, local change.

**Partial RoPE, verified**: `components/qwen`'s fixture currently sets
`ROPE_DIMENSION = HEAD_DIMENSION` (always equal), so today's one fixture
never exercises `dimension < head_dimension`. The review is nonetheless
correct that Magnetar's contract permits `rotary_dimension <= head_dimension`
generally (not fixture-specific), and this change's first-draft validation
(`head_count * dimension == cols`) would have rejected exactly that legal
case, in addition to computing the wrong per-head column offset
(`head * dimension` assumes no partial rotation).

**`lm_head`'s Host round-trip, verified**: `transpose_rows_cols`'s own doc
comment in `first_native_runtime.rs` states plainly why it exists: "matmul
... does not consult a `transpose_b` execution attribute" -- the portable
`matmul` Operator's shape rule validates `a[-1] == b[-2]` unconditionally,
with no `transpose_b`-aware alternative. The review's "recommended" option
(teach `matmul` about `transpose_b`) is a second, independent Operator
Core contract change on top of RoPE's; its own "acceptable alternative"
(pre-transpose once at Model Load, cache the result) achieves the same
hot-path outcome -- zero Host round-trips per generation step -- without
touching the `matmul` Operator contract at all. This design takes the
acceptable alternative, consistent with the review's own closing concern
about not shipping two independent Core contract changes in one pass.

**Weight shape provenance, verified**: `TensorEdge` (`execution_graph.rs`)
already carries a `descriptor: TensorDescriptor` field -- the graph's own
canonical shape source for every edge, including weight edges. This is
already in scope wherever a weight edge is resolved; the review's requested
fix is to use it, not invent a new metadata channel.

**`head_count`'s schema gap, verified**: `operator.rs`'s `"rope" =>
OperatorAttributeSchema::default().with_rule(...)` block lists exactly
`base`/`scale`/`dimension`/`position_mode`/`position_offset`.
`OperatorAttributeSchema::validate` rejects any attribute not in
`self.rules` (`"attribute is not defined by schema"`) -- confirmed by
reading the validation loop directly. Without adding `head_count` here,
every RoPE dispatch that sets it would fail validation before ever reaching
a Kernel.

**Affinity aggregation, verified**: `resource-affinity`'s "Affinity
Constraint Aggregation" requirement already exists and already defines the
right semantics in the abstract ("aggregate all resource affinities... SHALL
reject conflicting Provider, Device... bindings") -- this design reuses that
existing contract for Resident resources rather than inventing new
preserve/aggregate/reject rules.

## Goals / Non-Goals

**Goals:**
- RoPE's `head_count` is explicit graph data (set once, by the graph
  builder, from the architecture metadata it already has), not inferred at
  dispatch time from a node id's naming convention.
- RoPE correctly supports GQA/MQA (`kv_head_count != attention_head_count`)
  and partial rotation (`dimension <= head_width`, not `dimension == cols`
  divided evenly) -- both already-legal cases under existing contracts,
  neither newly invented here.
- `head_count` is validated the same way every other RoPE attribute is,
  through `operator.rs`'s real `OperatorAttributeSchema`, with the same
  reject-unknown-attribute guarantee every Operator already has.
- A weight bound to a Device-resident resource flows into its consuming
  Kernel without a forced Host download; its `NodeValue::Resident` shape
  comes from the graph's own `TensorEdge.descriptor`, never from a download
  or a recomputation.
- `lm_head`'s tied-embedding weight never round-trips through Host more
  than once, total, regardless of how many generation steps a session runs.
- `ResourceAffinity.provider`/`.device` recorded for every Qwen dispatch
  matches the Prepared Plan's actual resolved Provider/Device for freshly
  admitted resources, and preserves (via existing aggregation) a Resident
  resource's own recorded affinity instead of overwriting it -- enforced in
  production, not only in debug builds.
- MatMul -> RMSNorm and MatMul -> RoPE chains on the same Device consume the
  producer's output directly, zero Kernel invocations or Host round-trips
  between them beyond RoPE's own single, multi-head Kernel call.
- Pre-admission has a real transactional guarantee: a caller that pre-admits
  an output and then hits a submit-time or kernel/completion-time failure
  is left with no orphaned allocation and no falsely-Ready residency,
  proven for both failure points, not just asserted by one test.
- A real, hardware-verified first-native decode-shaped run through an
  actual `CudaProvider` (weights Device-resident, several chained CUDA
  Kernels, one final download) exists and passes before this change is
  considered complete.

**Non-Goals:**
- `ProviderConformanceProfile::Cuda`'s framework-level `Skipped`-vs-`Passed`
  distinction and `ProviderDataMovement`'s vacuous-advertisement gap: both
  are Conformance-framework concerns orthogonal to first-native's own
  dispatch correctness, tracked as an explicit, named follow-up (not folded
  in here, per the review's own allowance for this specific pair) --
  **not** treated as part of "the audit is closed" until that follow-up
  lands.
- Teaching the portable `matmul` Operator about `transpose_b`: deliberately
  not pursued (see Context); `lm_head`'s fix does not need it.
- A full generation loop (multiple decode steps, KV cache growth) through
  real CUDA hardware: the end-to-end proof required by this change is one
  representative forward pass (prefill-shaped), not a complete session --
  sufficient to prove the architecture is correct without building a second,
  separate long-running-session harness.

## Decisions

### Decision 1: `head_count` is explicit RoPE graph data, not inferred

`qwen_model_component.rs`'s RoPE node construction gains a `head_count`
attribute: the `rope_q` node gets `a.attention_head_count`, the `rope_k`
node gets `a.kv_head_count` (both already in scope there). `operator.rs`'s
`"rope"` schema gains:

```rust
.with_rule(
    "head_count",
    OperatorAttributeRule::optional(OperatorAttributeKind::Integer),
)
```

`first_native_runtime.rs`'s RoPE dispatch reads `head_count` via
`node_attribute_u64(node, "head_count")` when present, falling back to the
existing `qwen_rope_head_count(node, architecture)` id-suffix heuristic only
for a node that omits it (keeps any hand-written test graph that predates
this change working without modification; every graph this change's own
builder produces always sets it explicitly).

**Alternative considered**: keep inferring purely from the id suffix,
patched to also handle GQA (which it already does correctly). Rejected per
product's explicit decision: an id-naming convention is fragile
Runtime-side inference of information the graph builder already
authoritatively has; making it explicit graph data removes a whole class of
future bug (a differently-named RoPE node, or a future non-Qwen Model
Component, silently getting the wrong head count).

### Decision 2: `rope`'s multi-head semantics -- `head_width = cols / head_count`, partial RoPE preserved

Corrected from this change's first draft. Validation:

```text
head_count > 0
cols % head_count == 0
dimension > 0
dimension % 2 == 0
dimension <= head_width          where head_width = cols / head_count
```

(`head_count` omitted is treated as `1`, so `head_width == cols`,
reproducing today's exact existing validation
`dimension <= cols` for the single-block case.)

Per-head column offset: `col_base = head * head_width` (not `head *
dimension` -- the first draft's error, which only happens to coincide with
`head * head_width` when `dimension == head_width`, i.e. no partial
rotation). Rotation touches exactly `[col_base, col_base + dimension)`;
columns `[col_base + dimension, col_base + head_width)` within a head (the
partial-RoPE tail) and any columns beyond `head_count * head_width` (none,
given the `cols % head_count == 0` invariant) are left untouched, copied
from input to output unchanged -- **not** zeroed, correcting the first
draft's `alloc_zeros`-then-only-write-rotated-columns behavior, which was
silently correct only because every existing call always has
`dimension == cols` today.

CUDA-side (`kernels.cu`'s `rope_kernel`), the parallelization axis is
`(row, head, pair)` where `pair` ranges over `dimension / 2` (not
`head_width / 2` -- only the rotated sub-range needs a thread):

```text
half  = dimension / 2
total = rows * head_count * half
row   = idx / (head_count * half)
rem   = idx % (head_count * half)
head  = rem / half
pair  = rem % half
col_base  = head * head_width
position  = (position_offset + row) * scale        // unchanged: positional, not per-head
frequency = base^(-2*pair/dimension)                // unchanged, dimension is the per-head rotation width
out[row, col_base + 2*pair]   = even*cos - odd*sin
out[row, col_base + 2*pair+1] = even*sin + odd*cos
```

The kernel's output buffer is seeded as a copy of the input (not
zero-allocated) before the rotation writes, so untouched columns (partial
RoPE's tail, any head padding) carry the original values through rather
than zeros. `head_count == 1, dimension == cols` (today's only exercised
case) reproduces the exact existing formula and output bit-for-bit.

`providers/cpu::rope` gains the identical parameter and the same
`head_width`/`col_base` semantics, looped in Rust.

**Alternative considered**: the first draft's `head_count * dimension ==
cols`. Rejected: breaks GQA (`kv_head_count` heads times a `dimension`
sized for `attention_head_count`'s head width need not equal `cols` for K)
and breaks partial RoPE (`dimension < head_width` is legal and already
implied by existing per-head-slice call sites, which never validated the
equality explicitly -- they simply always passed `dimension == head_width`
by construction, so the gap was latent, not exercised).

### Decision 3: Weight edges thread `NodeValue`, shaped from `TensorEdge.descriptor`

`resolve_qwen_weight_edge`'s return type changes `HostTensor` -> `NodeValue`.
It already resolves `resource_id` from `weight_bindings`; it additionally
resolves that edge's `TensorDescriptor` from the graph (`TensorEdge.
descriptor`, already a field on every edge including weight edges -- the
canonical shape source, not the Provider's storage and not Qwen's own
config). `provider.read_tensor_value(resource_id)` maps to:

```text
TensorValue::Host(tensor)  -> NodeValue::Host(tensor)
TensorValue::Opaque        -> NodeValue::Resident { id: resource_id, shape: edge.descriptor.shape.dimensions.clone() }
```

mirroring `dispatch_reference_cpu_operator_multi`'s own existing
`TensorValue` -> `NodeValue` mapping exactly, just resolving `Opaque`'s
shape from the edge instead of from a `TensorResourceDescriptor` the
dispatch already had in hand there. The `lm_head`/tied-embeddings case (see
Decision 5) stops needing a Rust-side transform on every call, so it no
longer forces `.into_host()` in the hot path either -- it resolves its own
(already-transposed) resource id directly, the same as any other weight.

**Alternative considered**: derive shape by downloading the weight once and
caching its shape. Rejected per product's explicit decision: the shape is
already known, statically, from the graph -- downloading data to learn
metadata the graph already carries is unnecessary work and a second,
redundant source of truth that could drift from the graph's own descriptor.

### Decision 4: RMSNorm drops its Rust-side broadcast, becomes `NodeValue`-based

Unchanged from the first draft (the review raised no objection to this
part): `dispatch_qwen_rmsnorm`'s signature changes `input: HostTensor,
weight: HostTensor` -> `input: NodeValue, weight: NodeValue`; the manual
per-row broadcast loop is deleted -- both real Kernels already accept
`[cols]` and broadcast internally, verified directly in
`providers/cpu::rmsnorm`/`providers/cuda::CudaKernels::rmsnorm`.

### Decision 5: `lm_head` tied embeddings transpose once, at Model Load

`WeightMaterializationTransaction::stage_weight` (the existing Model-Load-
time weight staging path, `first_native_runtime.rs`) gains a
tied-embeddings-specific step: after staging `token_embedding`, if
`tied_embeddings` is set, also transpose it once
(`transpose_rows_cols`, the existing helper, unchanged) and stage the
result under its own resource id (e.g. `token_embedding.transposed`) through
the same admission path every other weight uses -- a genuine Device-resident
resource from that point on, not a per-call Rust computation.
`resolve_qwen_weight_edge`'s `lm_head` branch changes from "download,
transpose in Rust, return" to "resolve the pre-transposed resource id
directly", following Decision 3's `NodeValue` path like any other weight --
no special-cased materialization left in the hot path.

**Alternative considered** (the review's own "recommended" option): give
`matmul` a `transpose_b` attribute and have its shape rule validate
`a[-1] == b[-1]` instead of `a[-1] == b[-2]` when set, so `lm_head` never
needs its own transposed copy at all. Not pursued here: a second,
independent Operator Core contract change (touching `operator.rs`'s
`matmul` shape rule, both Providers' Kernel dispatch of `transpose_b`, and
the `operator`/`operator-scope` capability specs again) beyond RoPE's own,
which the review's closing paragraph specifically warned against
attempting simultaneously. Left as a genuinely reasonable future
simplification if a *second* transpose-avoidance need ever arises
elsewhere; not required to close this change's scope.

### Decision 6: `ResourceAffinity` -- fresh derives from the Plan, Resident is preserved and aggregated

`dispatch_reference_cpu_operator_multi`'s per-resource affinity handling
splits by whether the resource is Fresh or already Resident:

```text
NodeInputResource::Fresh(..)     -> affinity = resolved_resource_affinity(prepared_plan, node, execution_context)
NodeInputResource::Resident(id, ..) -> affinity = aggregate(existing_residency.affinity, resolved_resource_affinity(...))
```

using `resource-affinity`'s own existing "Affinity Constraint Aggregation"
contract for the `Resident` branch (reject on conflict, preserve distinct
non-conflicting bindings) -- not a new, first-native-specific
preserve/merge rule. Output resources (this dispatch's own new admissions)
always take the Fresh path: `resolved_resource_affinity` (new helper,
mirroring `resolved_output_placement`'s exact lookup pattern, described
identically to the first draft) derived from the Prepared Plan binding.

**Production validation** (not `debug_assert!`): after
`KernelDispatchPlan` is built, a structured check --
`InferenceApiError`/`ProviderExecutionError`-shaped, not a panic -- verifies
the resolved invocation's Provider/Device agrees with the affinity attached
to its resources, for every dispatch, in every build. This lives at the
generic Kernel/Runtime dispatch boundary (alongside
`validate_invocation`'s existing memory-class/layout/dtype checks, the same
call site), not first-native-specific, so any future caller of the same
dispatch machinery inherits the check automatically.

**Alternative considered** (this change's first draft): `debug_assert!`
only. Rejected per product's explicit decision: a class of bug real enough
to have shipped once (this change's own root cause) deserves a production
guarantee, not only a development-time tripwire that a release build
silently skips.

### Decision 7: Pre-admission rollback is a transactional guarantee, not one test

Formalized as explicit rollback behavior, verified at both failure points
pre-admission can hit:

- **Submit-time failure** (the Kernel Registry/dispatch plan construction
  fails after `MemoryManager::admit_kernel_output` already ran): the
  pre-admitted allocation is released and its `TensorResidency` record
  removed -- the caller's admission is undone, not left dangling.
- **Kernel/completion-time failure** (dispatch plan built, `submit_kernel`
  itself returns an error or the Provider's own completion reports
  failure): same guarantee -- no orphaned allocation, no residency record
  implying data that was never actually written, and the Provider's own
  storage (its `resource_allocations` tracking) is not left believing it
  self-admitted something it did not.

Both paths reuse the existing `TestFailableProvider`-style failure
injection this test suite already has for other rollback proofs, rather
than a new injection mechanism.

**Alternative considered**: keep this as a single test proving the happy
rollback path exists. Rejected per product's explicit decision: a
transactional guarantee needs both failure points independently verified,
not one representative case standing in for both.

### Decision 8: A real CUDA end-to-end run is a required exit criterion

A new test (or small test fixture), real hardware, `arc-gpu-magnetar`:
registers an actual `CudaProvider`, resolves a `ModelInstance`/Prepared Plan
onto it, stages weights Device-resident, and dispatches a representative
forward-pass-shaped chain (weight -> MatMul -> RMSNorm -> RoPE -> a
projection) verifying: no `ResidencyUnavailable`, no D2H/H2D between
compatible chained Kernels beyond what the chain's own final download
requires, `ResourceAffinity`/`MemoryPlacement` both correctly reflect CUDA/
GPU0 throughout, and the numeric result agrees with Reference CPU within
tolerance. Scoped as one representative forward pass (see Non-Goals), not a
full multi-step generation session.

**Alternative considered**: keep this deferred as a documented follow-up
(this change's first draft). Rejected per product's explicit decision: the
whole point of this change is to make first-native genuinely CUDA-capable;
shipping the mechanism without ever proving it end-to-end on real hardware
leaves exactly the same "never actually exercised" gap the previous change
was honest about, and the review does not accept that gap being carried
forward again.

## Risks / Trade-offs

- [`rope`'s signature change touches two Provider crates' public free
  functions] -> both are `publish = false`, submodule-only crates this
  change also updates in the same session; no external consumer exists
  outside this repository's own pinned commits.
- [CUDA kernel indexing change is easy to get subtly wrong] -> `head_count
  == 1, dimension == cols` must reproduce today's exact kernel behavior
  bit-for-bit against the existing `rope_matches_reference_cpu` conformance
  test; new conformance cases specifically for GQA-shaped head_count and
  for partial RoPE (`dimension < head_width`), each comparing CUDA against
  `providers/cpu`'s own corrected implementation, on real hardware.
- [Task 8's new E2E harness could grow into disproportionate scaffolding,
  the exact concern that descoped it from the previous change] -> scoped
  tightly to one representative forward pass reusing as much of the
  existing `TestFailableProvider`/Prepared Plan test infrastructure as
  possible; if it turns out to need materially more than that, stop and
  report the size back before continuing rather than open-endedly
  expanding it.
- [Decision 6's production validation adds a new check to a hot dispatch
  path] -> cost is one comparison of already-computed values (Provider/
  Device identifiers), not new I/O or allocation; negligible relative to a
  Kernel dispatch itself.
- [Removing `dispatch_qwen_rope_per_head` changes per-node Kernel dispatch
  count for causal-evidence observers] -> re-verify against the existing
  per-node causal-evidence chain tests (`reach-architecture-freeze-1` task
  group 17), unchanged from the first draft's own risk note.

## Migration Plan

1. Land Decision 6 (ResourceAffinity: resolved helper + Resident
   preservation + production validation) first, in isolation -- a
   conformance fix against an already-existing spec requirement, no new
   Kernel surface.
2. Land Decision 4 (RMSNorm) next -- same category, no new Kernel surface.
3. Land Decisions 1-2 (RoPE `head_count`, corrected semantics) together --
   they are one coherent unit (the attribute and its validation cannot be
   separated meaningfully) -- verified on real CUDA hardware before
   touching `first_native_runtime.rs`'s RoPE dispatch.
4. Land Decision 3 (weight edges) and Decision 5 (`lm_head` pre-transpose)
   together -- Decision 5 depends on Decision 3's `NodeValue` path existing
   first.
5. Land Decision 7 (rollback) in parallel with 1-4; additive, no ordering
   dependency.
6. Land Decision 8 (E2E CUDA test) last, after 1-5 are all in place and
   individually verified -- it is the integration proof for everything
   above, not a substitute for verifying each decision on its own first.
7. Update `providers/cpu`/`providers/cuda` submodule pins together with the
   parent commit, matching this session's established sequencing.
8. Dispatch `gpu-runner-smoke.yml`, confirm green on the exact shipped
   commits.

## Open Questions

(none outstanding -- all decisions above reflect explicit product review
feedback, not open forks.)
