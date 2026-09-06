## Context

`docs/audits/cuda-provider-audit-synthese-2026-09-06.md` approved the CUDA
Provider standalone (`Magnetar-provider-CUDA@8eb0160`) but found first-native
(`magnetar-runtime/src/first_native_runtime.rs`) still forces Host round-trips
in three places once a real Device-resident Provider is actually selected.
Each of the three P0s was re-verified directly against the code (not assumed
from the audit's own description) before this design was written:

- **Weight edges**: `resolve_qwen_weight_edge` calls
  `provider.read_tensor_value(resource_id)` then unconditionally
  `TensorValue::into_host(resource_id)`. A CUDA-resident weight returns
  `TensorValue::Opaque`, and `into_host` on `Opaque` is a structured error
  (`ResidencyUnavailable`) by design (`generalize-first-native-provider-
  dispatch`) -- so this is a real failure path today, not a hypothetical one,
  the moment `ModelInstance` placement ever resolves a weight onto CUDA.
- **ResourceAffinity**: `dispatch_reference_cpu_operator_multi` (the one
  dispatch helper every Qwen operator funnels through) builds
  `ResourceAffinity::new(...).with_provider(ProviderBinding::new(
  REFERENCE_CPU_PROVIDER_NAME))` unconditionally, two lines above
  `resolved_kernel_memory_class(ctx.prepared_plan.as_deref(), &node)`, which
  *does* correctly derive `KernelMemoryClass::Device` from the same Prepared
  Plan binding. The same function produces two different, contradictory
  answers to "which Provider is this for" a few lines apart.
- **RMSNorm/RoPE**: both take `HostTensor` (not `NodeValue`) as their input
  parameter, so `dispatch_qwen_graph_node`'s "rmsnorm"/"rope" arms call
  `.into_host()` on their input before dispatch, unconditionally.

Two of these three are not new requirements to invent: `device-resident-
resource`'s "Same-Device Pipeline Avoids Mandatory Host Copy" already has a
scenario literally named "MatMul to RMSNorm" requiring no host round-trip,
and `resource-affinity`'s "Runtime-Native Resource Affinity" already requires
"the resource affinity records the selected Provider" -- first-native's
current code does not conform to specs that already exist. Only RoPE's fix
needs new capability surface (a Kernel-level `head_count`), which is why it
is the one item in this change with a real design decision behind it rather
than a straightforward conformance fix.

Investigation during this design pass found the actual root causes are
narrower than they first look:

- `providers/cpu::rmsnorm`/`providers/cuda::CudaKernels::rmsnorm` both derive
  `cols` from the input's own shape and accept a `[cols]`-shaped weight
  directly -- `dispatch_qwen_rmsnorm`'s manual per-row broadcast in Rust
  (`first_native_runtime.rs`, building an `[rows, cols]` weight before
  dispatch) is pure duplication of work each Kernel already does internally.
  Removing it is a correctness-neutral simplification, not a behavior
  change to what the Kernel computes.
- `providers/cpu::rope`/`providers/cuda::CudaKernels::rope` both only rotate
  the first `dimension` columns of every row using consecutive-pair
  rotation with `position = position_offset + row`. `dispatch_qwen_rope_per_
  head`'s Rust-side per-head slice-dispatch-reassemble loop exists *because*
  today's Kernel genuinely cannot rotate more than one head-sized block per
  call -- this is a real Kernel-scope limit, not orchestration laziness, and
  was flagged to product as a genuine two-way fork (extend the Kernel now,
  vs. keep the Host materialization as a documented, permanently narrow
  exception since it only round-trips Q/K, not the full hidden state).
  Product decided: extend the Kernel.

## Goals / Non-Goals

**Goals:**
- A weight bound to a Device-resident resource flows into its consuming
  Kernel without a forced Host download, while `lm_head`'s tied-embedding
  transpose (a genuine Rust-side data transform) keeps working exactly as
  before.
- `ResourceAffinity.provider`/`.device` recorded for every Qwen dispatch
  matches the Prepared Plan's actual resolved Provider/Device, not a
  hardcoded Reference CPU identity -- verified by a debug-time cross-check,
  not just by inspection.
- MatMul -> RMSNorm and MatMul -> RoPE chains on the same Device consume the
  producer's output directly, with zero Kernel invocations or Host
  round-trips between them beyond what RoPE's own (now single, multi-head)
  Kernel call performs.
- The `rope` Kernel Operator's `head_count` extension is backward compatible:
  every existing caller (conformance tests, any direct dispatcher) that
  never sets it keeps today's exact single-block behavior.
- A regression test proves the pre-admission mechanism
  (`unify-provider-output-admission-and-residency`) rolls back cleanly when
  the Kernel dispatch itself fails after a caller already pre-admitted the
  output -- no orphaned allocation, no falsely-Ready residency record.

**Non-Goals:**
- A literal, hardware-verified, full first-native decode step through a real
  `CudaProvider` end to end (still separate, larger work: needs a
  CUDA-registered `Runtime`/`ModelInstance` fixture that does not exist
  yet). This change makes that path *architecturally correct*; proving it
  end-to-end on real hardware is tracked as a follow-up, not blocked on
  anything here.
- `ProviderConformanceProfile::Cuda`'s framework-level `Skipped`-vs-`Passed`
  distinction (audit P1-1) and `ProviderDataMovement`'s vacuous-advertisement
  gap (audit P1-2): both are Conformance-framework concerns orthogonal to
  first-native's own dispatch correctness, deferred to their own follow-up
  rather than folded into an already cross-cutting change.
- Attention: already `NodeValue`-based end to end (`dispatch_qwen_attention`
  takes `q`/`k`/`v: NodeValue`); the audit did not flag it and re-reading it
  during this pass found no forced materialization to fix.

## Decisions

### Decision 1: Weight edges thread `NodeValue` through, materialize only for the tied-embedding transpose

`resolve_qwen_weight_edge` changes its return type from `HostTensor` to
`NodeValue`. Its body: same resource-id lookup as today, but
`provider.read_tensor_value(resource_id)` maps directly to `NodeValue::Host`/
`NodeValue::Resident` (mirroring `dispatch_reference_cpu_operator_multi`'s
own existing `TensorValue` -> `NodeValue` mapping) instead of immediately
calling `.into_host()`. The `lm_head`/tied-embeddings transpose is the one
case that still needs real bytes (`transpose_rows_cols` is a genuine
Rust-side reshape): that branch calls `.into_host()` itself, scoped to
exactly the case that needs it, rather than the function doing it
unconditionally for every weight.

**Alternative considered**: keep `resolve_qwen_weight_edge` returning
`HostTensor` and instead make the *caller* re-upload a downloaded weight
under a new Resident id before dispatch. Rejected: this is exactly the
round-trip the audit flags, just moved one call frame up -- no different
from today's behavior, only relocated.

### Decision 2: `resolved_resource_affinity` replaces the hardcoded Provider binding

New helper, mirroring `resolved_output_placement`'s existing lookup exactly:

```rust
fn resolved_resource_affinity(
    prepared_plan: Option<&PreparedExecutionPlan>,
    node: &ExecutionNodeId,
    execution_context: ExecutionContextId,
) -> ResourceAffinity
```

Returns `ResourceAffinity::new(FallbackClass::Transparent)
.with_provider(binding.provider.clone())
.with_device(binding.device.clone())` (device omitted when the binding has
none, exactly as `resolved_output_placement` already falls back) when a
Prepared Plan binding exists for `node`, and the existing Reference-CPU
default otherwise (no Prepared Plan is the case every existing direct-
dispatch/test caller without a Plan already exercises, so its behavior does
not change). `dispatch_reference_cpu_operator_multi` calls this once, in
place of its current hardcoded construction, and reuses the *same* affinity
value for every input/output resource in that dispatch -- it already does
this today, only the value's source changes.

**Cross-check** (debug-time, not a new production error path): after
`KernelDispatchPlan` is built, assert (`debug_assert!`, so it costs nothing
in release builds and cannot itself become a new production failure mode)
that `plan.invocation`'s resolved Provider/Device binding
(`selection`/`advertisement`, already in scope) agrees with `affinity`'s
Provider/Device. This exists to catch a *future* regression reintroducing
the same class of bug (a helper drifting out of sync with
`resolved_kernel_memory_class`/`resolved_output_placement`), not because
production code should ever hit it once Decision 2 lands correctly.

**Alternative considered**: a hard, always-on
`InferenceApiError`-returning validation instead of `debug_assert!`.
Rejected: `validate_invocation`'s existing memory-class/layout/dtype checks
are the right place for genuine runtime-data-dependent validation; this
specific check is a static consistency property of this module's own two
helper functions agreeing with each other, exactly the kind of invariant a
debug assertion exists for.

### Decision 3: RMSNorm drops its Rust-side broadcast, becomes `NodeValue`-based

`dispatch_qwen_rmsnorm`'s signature changes `input: HostTensor, weight:
HostTensor` -> `input: NodeValue, weight: NodeValue`; the manual per-row
broadcast loop (building an `[rows, cols]` copy of the weight) is deleted --
both real Kernels already accept `[cols]` and broadcast internally, verified
directly in `providers/cpu::rmsnorm`/`providers/cuda::CudaKernels::rmsnorm`.
Its two inputs become `NodeInputResource::Resident`-eligible in
`dispatch_reference_cpu_operator_multi`'s call, exactly like matmul's
already do. The "rmsnorm" arm in `dispatch_qwen_graph_node` stops calling
`.into_host()` on either input before dispatching.

**Alternative considered**: keep the broadcast but make it operate on
`NodeValue` lazily (only materializing if a broadcast is actually needed).
Rejected once confirmed unnecessary: no real Kernel needs the broadcast at
all, so there is nothing left to make lazy -- deleting it is strictly
simpler than any lazy variant.

### Decision 4: `rope` Kernel Operator gains a native `head_count`

`providers/cpu::rope`/`providers/cuda::CudaKernels::rope` gain a new
trailing parameter `head_count: u64` (`1` reproduces today's exact
behavior bit-for-bit -- this is the regression guard both Providers'
existing single-head tests continue to exercise unchanged). Validation
extends to require `head_count >= 1` and `head_count * dimension == cols`
(replacing today's implicit, unvalidated assumption that a caller always
passes `dimension == cols`) -- `dimension` keeps its existing name and
means "rotation width per head", not renamed to `head_dimension`, since
every existing call site already passes it that way and this avoids
renaming a stable Kernel attribute for no behavioral reason.

Rust-side (`providers/cpu`) loops one extra level (`for head in
0..head_count`) around the existing per-row-pair loop, rotating columns
`[head * dimension, head * dimension + dimension)` per row per head instead
of always `[0, dimension)`.

CUDA-side (`kernels.cu`'s `rope_kernel`), the parallelization axis extends
from `(row, pair)` to `(row, head, pair)`:

```text
half = dimension / 2                    // per-head pair count, unchanged
total = rows * head_count * half        // was rows * half
row   = idx / (head_count * half)
rem   = idx % (head_count * half)
head  = rem / half
pair  = rem % half
col_base = head * dimension             // was implicitly 0
position  = (position_offset + row) * scale        // unchanged: positional, not per-head
frequency = base^(-2*pair/dimension)                // unchanged, dimension is per-head width
out[row, col_base + 2*pair]   = even*cos - odd*sin  // was out[row, 2*pair]
out[row, col_base + 2*pair+1] = even*sin + odd*cos  // was out[row, 2*pair+1]
```

`head_count == 1` collapses `col_base` to `0` and reproduces today's exact
indexing -- confirmed by construction, not just by the formula's shape.
`CudaExecutor`'s "rope" dispatch arm reads a new `head_count` attribute
(`Self::attribute_u64(&invocation.attributes, "head_count", 1)`, default
`1` so existing callers/tests that never set it are unaffected).

`first_native_runtime.rs`: `dispatch_qwen_rope_per_head` is deleted;
replaced by a single dispatch per RoPE node passing `head_count` (from the
architecture metadata already available: `attention_head_count`) and the
full, un-sliced `NodeValue` input straight through -- one Kernel invocation
per RoPE node instead of `head_count` of them, and eligible for
`NodeInputResource::Resident` like every other operator.

**Alternative considered** (the one presented to product as a genuine
fork): keep `dispatch_qwen_rope_per_head`'s Rust-side per-head
materialization permanently, documented as a narrow, accepted exception
(only Q/K round-trip, not the full hidden state MatMul/RMSNorm exchange).
Rejected by product decision: extend the Kernel now, before more Providers
(Metal/OpenVINO/QNN/WebGPU) each have to either reimplement the same
per-head Rust decomposition or independently decide to support
`head_count` themselves later -- same reasoning as `unify-provider-output-
admission-and-residency`'s "fix generically before more Providers exist".

### Decision 5: Pre-admission rollback regression test (audit P1-4)

New test (home: wherever `admit_kernel_output_replaces_and_releases_the_
previous_allocation_for_the_same_id`/`reference_cpu_honors_a_caller_pre_
admitted_output_without_double_admitting` already live,
`magnetar-runtime/src/tests.rs`): pre-admit an output via
`MemoryManager::admit_kernel_output`, then dispatch a Kernel invocation
constructed to fail during execution (reusing the existing
`TestFailableProvider`-style failure injection this test suite already has
for other rollback proofs), and assert afterward: the Memory Manager holds
exactly the allocation count it held before the failed attempt (no orphan),
and the pre-admitted id's `TensorResidency` is not left in a `Ready`/`Active`
state implying data that was never actually written.

### Decision 6: `implement-cuda-provider-baseline/tasks.md` accuracy pass (audit P1-3)

Read through and correct task descriptions that still describe pre-
`enable-device-resident-kernel-chaining` behavior (Host-resident storage,
per-kernel-forced round-trip) now that the Provider is genuinely
Device-resident -- a documentation-only pass, no task re-scoping, done
last (after Decisions 1-4 land) so it describes the actual final state
rather than an intermediate one.

## Risks / Trade-offs

- [`rope`'s signature change touches two Provider crates' public free
  functions] -> both are `publish = false`, submodule-only crates this
  change also updates in the same session; no external consumer exists
  outside this repository's own pinned commits.
- [CUDA kernel indexing change is easy to get subtly wrong (off-by-one in
  `col_base`, wrong axis order)] -> `head_count == 1` must reproduce
  today's exact kernel behavior bit-for-bit, verified against the existing
  `rope_matches_reference_cpu` conformance test unchanged, plus a new
  multi-head conformance case comparing CUDA's multi-head output against
  Reference CPU's own multi-head loop for the same input, on real hardware.
- [Removing `dispatch_qwen_rope_per_head` changes per-node Kernel dispatch
  count (was `head_count` invocations, becomes 1) for causal-evidence
  observers] -> re-verify against the existing per-node causal-evidence
  chain tests (`reach-architecture-freeze-1` task group 17), same
  precedent as `unify-provider-output-admission-and-residency`'s Decision
  3 risk.
- [Decision 2's `debug_assert!` never fires in this session's own
  release-mode CI/CD verification, so a regression could still slip into a
  release build undetected] -> accepted: this specific class of bug
  (two helpers disagreeing) is exactly what code review and this change's
  own test suite additions are for; the assertion is a development-time
  tripwire, not the only line of defense.

## Migration Plan

1. Land Decision 2 (`resolved_resource_affinity`) and Decision 3 (RMSNorm)
   first, in isolation -- both are conformance fixes against already-
   existing spec requirements, no new Kernel surface, lowest risk.
2. Land Decision 1 (weight edges) next -- depends on nothing else here, but
   ordered after 2/3 so the full regression suite already covers the
   affinity/RMSNorm changes before adding a third simultaneous change.
3. Land Decision 4 (RoPE `head_count`) last and separately verified on real
   CUDA hardware (`providers/cuda`'s own test suite, plus the conformance
   suite) before touching `first_native_runtime.rs`'s RoPE dispatch --
   this is the one decision with real Kernel-correctness risk.
4. Land Decision 5 (rollback test) and Decision 6 (tasks.md accuracy) in
   parallel with 1-4; both are additive/documentation, no ordering
   dependency.
5. Update `providers/cpu`/`providers/cuda` submodule pins together with the
   parent commit, matching this session's established sequencing.
6. Dispatch `gpu-runner-smoke.yml`, confirm green on the exact shipped
   commits, same as every prior change this session.

## Open Questions

(none outstanding -- the one genuine fork, RoPE's fix approach, was
resolved by product decision before this design was written; see Decision
4's "Alternative considered".)
