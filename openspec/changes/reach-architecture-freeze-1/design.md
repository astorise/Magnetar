## Context

The post-audit review that produced `apply-post-audit-first-native-correctifs`
was addressed by `d9d26f7` ("Introduce first-native datapath authority and
execution improvements"). A second audit pass (2026-09-01), run against that
result, found the first-native vertical slice still has causal gaps between
what the specs already require and what the code does:

- Provider output/workspace/KV bytes can still be materialized before Runtime
  memory admission succeeds.
- A `ProviderExecutionHandle` can be constructed after the numerical Kernel
  work already ran, so `complete()` is evidence of nothing.
- `first-native` resolves a `dyn Provider` and downcasts it to
  `ReferenceCpuProvider` to reach `ReferenceCpuExecutor`, which blocks any
  second Provider.
- A parallel helper turns a published `PlanNodeBinding` back into a synthetic
  `KernelCandidate` instead of driving execution through
  `PreparedExecutionPlanExecutor`.
- The generic graph executor threads `HostTensor` between nodes, which does
  not represent device-only or multi-output intermediates.
- Qwen weights reach the compute path through a fixture helper
  (`bind_qwen_fixture_weights()`) called after `load_model()`, not through
  `Model Loading` reading `Model Artifact` bytes.
- Strict first-native can still fall back to a Rust-synthesized Qwen graph
  when no Component Engine is available.
- `magnetar-runtime` still owns Qwen-specific graph-building and weight/KV
  naming logic, which blocks adding `components/llama` without touching the
  Core.
- KV pending→commit is logically transactional but Provider bytes can be
  written before admission, and there is no clear release primitive for
  abandoned pending resources.
- `components/qwen`, `components/llama`, `formats/gguf`,
  `formats/safetensors` were declared in `.gitmodules` intent but not
  materialized as gitlinks (this branch has since begun that work directly).

`openspec/changes/apply-post-audit-first-native-correctifs` and
`openspec/changes/make-first-native-datapath-authoritative` cover the prior
audit round; this Change is scoped to the delta the second audit found.

## Goals / Non-Goals

**Goals:**
- Define the one contract genuinely missing from the spec set: how a Model
  Component exports portable graph semantics to the Runtime
  (`model-component-graph-contract`), so `qwen_prefill_graph` /
  `qwen_decode_graph` can be retired from `magnetar-runtime` without losing
  graph production entirely.
- Track every other audit finding as an implementation task against a spec
  that is already correct, with the existing requirement it must satisfy
  named explicitly, so `tasks.md` doubles as the punch list for Architecture
  Freeze #1 (`first-native-implementation-cut`'s "Architecture Freeze #1"
  requirement).
- Keep the phased dependency order the audit derived (memory/causality before
  plan authority before model data authority before KV transactionality
  before the Component graph contract before Provider extraction) so partial
  progress stays coherent instead of racing independent fixes.

**Non-Goals:**
- Re-litigating requirements that already exist and are already correct
  (`memory`, `provider`, `kernel-execution-plan`, `kernel-registry`,
  `execution-graph`, `model-loading`, `kv-cache`, `runtime`). This Change
  does not touch their requirement text.
- Defining `define-provider-prepared-kernel-execution-contract`. The audit
  flags this as conditional on whether `ProviderExecutionApi`'s existing
  `ComputeExecutionPlan`-centric payload can carry `PreparedPlanNodeExecution`
  + `TensorResourceId` work without contortions. That original question was
  answered without needing it (see "Open Questions" below) -- but a
  *different* question, found later, did need it, and it now exists as a
  real Change directory; see that same Open Questions entry for the
  distinction.
- Extracting `providers/cpu` as its own crate/submodule (audit Correctif 15).
  That depends on the Provider-downcast removal and causal-handle work
  landing first, and is tracked as a follow-up task, not designed here.
- Cryptographic artifact signatures (audit Correctif 19). Tracked as a
  future issue per the audit; out of scope here.

## Decisions

### The Model Component graph contract is Runtime-owned (Option B), not a serialized descriptor (Option A)

The audit presented both options neutrally. Runtime-owned graph-builder
capability wins because:
- It reuses `magnetar-runtime`'s existing `ExecutionGraph`,
  `TensorDescriptor`, and validation types instead of requiring a second,
  independently-versioned serialization format that must be kept in sync.
- It avoids opaque JSON/CBOR blobs crossing the WIT boundary, which would
  need their own schema evolution story on top of WIT's.
- It structurally prevents a Component from supplying fields the Runtime
  must recompute anyway (e.g. Provider/Device binding), because the builder
  API only exposes operations the Runtime is willing to accept.
- It matches the existing architectural invariant that Components request
  Capabilities and the Runtime owns resolution — a builder Capability is a
  natural fit; a self-describing blob is not.

Trade-off: the builder Capability surface must be designed carefully (it is
effectively a new WIT interface), and versioning happens through Capability
versioning rather than a single descriptor version field. This is judged
acceptable because `capability`'s existing `Capability Versioning`
requirement already covers that mechanism.

### One Change, not three, for this audit round

The audit's own section 26 lists up to three new Changes (Change A
mandatory, B conditional, C optional) plus 16 implementation issues. Rather
than open a separate Change per audit finding, this Change:
- Defines the one new capability (Change A equivalent:
  `model-component-graph-contract`).
- Carries every implementation-only finding as a task with an explicit
  Definition of Done, phased per the audit's section 21 ordering.

Rationale: the implementation tasks do not change any requirement text, so
they do not need their own spec-delta artifacts — creating a Change per item
would produce empty or duplicate "Modified Capabilities" sections and
fragment a single coherent freeze effort across a dozen trackers. Change B
(`define-provider-prepared-kernel-execution-contract`) and Change C
(`externalize-runtime-extension-modules`) remain explicitly out of scope
(see Non-Goals) and should be opened separately if and when they turn out to
be needed.

### No spec deltas for correctifs 1-7, 9, 11-17

Per this repo's OpenSpec governance rule (`spec already correct + code not
conformant` → implementation issue; `new semantic decision` → new Change),
and having re-read the current requirement text for `memory`, `provider`,
`kernel-execution-plan`, `kernel-registry`, `execution-graph`,
`model-loading`, `kv-cache`, and `runtime` while preparing this design, the
existing requirements already say what the audit wants (for example
`memory`'s "Memory Baseline Precedes Provider Execution" and "Memory
Admission", `provider`'s "Provider Execution Handle" and "Completion
Result"). Changing their text would be spec churn with no semantic delta;
the gap is purely in `magnetar-runtime`'s implementation.

## Risks / Trade-offs

- [Risk] Treating most findings as "implementation-only" could hide a real
  spec gap if the current requirement text turns out to be ambiguous once
  implementation starts. → Mitigation: `tasks.md` requires each task to cite
  the exact existing requirement name it satisfies; if implementation proves
  a requirement insufficient, that item is pulled into its own Change instead
  of being patched silently here.
- [Risk] The Runtime-owned graph-builder Capability (Option B) is a new WIT
  surface with real design cost (interface shape, error mapping, versioning)
  that the audit did not fully specify. → Mitigation: `specs/model-component-graph-contract/spec.md`
  defines the required semantics and scenarios; exact WIT signatures are
  implementation detail resolved during `tasks.md` execution, consistent with
  how other Capability contracts in this repo are specified.
- [Risk] Retiring `qwen_prefill_graph`/`qwen_decode_graph` from the
  production path is **BREAKING** for anything still depending on the
  Rust-synthesized graph (fixtures, tests). → Mitigation: keep the Rust
  builder available under explicit test/conformance support only, mirroring
  how `apply-post-audit-first-native-correctifs` handled the synthetic-logits
  removal.
- [Trade-off] Bundling 16 implementation issues into one Change's `tasks.md`
  makes that file large and long-lived. Accepted because the audit's own
  phase ordering (section 21) treats them as one coherent effort gating a
  single outcome (Architecture Freeze #1 acceptance), and splitting them
  across trackers would lose that ordering.

## Migration Plan

Phased per the audit (section 21), each phase gated on the previous:

1. **Memory + Provider causality**: admission before materialization, causal
   Provider submit/complete, remove concrete Provider downcasts.
2. **Prepared plan executor authority**: production execution goes through
   `PreparedExecutionPlanExecutor`; canonicalize `ExecutionGraph` topology;
   remove `std::mem::take(MemoryManager)`.
3. **Model data authority**: `Model Loading` creates the weight resources
   execution actually consumes; remove the Qwen fixture side-channel.
4. **KV transaction**: reserve → prepare → materialize → commit/abort →
   release, atomic across Runtime and Provider.
5. **Model Component graph contract**: implement
   `model-component-graph-contract`; make strict first-native fail closed
   without a compatible Component; remove Qwen-specific semantics from
   `magnetar-runtime`.
6. **External Provider boundary** (conditional): only if Phase 1's causal
   `ProviderExecutionApi` payload cannot cleanly carry
   `PreparedPlanNodeExecution` + `TensorResourceId` work, open
   `define-provider-prepared-kernel-execution-contract` and extract
   `providers/cpu`.
7. **Submodule / integration CI**: lock `components/qwen`,
   `components/llama`, `formats/gguf`, `formats/safetensors` to pinned
   commits (gitlinks already added on this branch); add tiered CI (Core /
   Component / Format / Provider / Full conformance) without making the Core
   clone depend on any submodule.
8. **Governance close-out**: reconcile the OpenSpec completion tracker and
   README with actual causal status; file the cryptographic-signature and
   CI-pinning follow-up issues.

Rollback: each phase is independently revertible in git since none of them
change public spec requirement text except the new
`model-component-graph-contract` capability, which is additive (no existing
Component is required to implement it until the Qwen Component migrates in
Phase 5).

## Open Questions

- ~~Does `ProviderExecutionApi`'s existing payload shape survive carrying real
  `PreparedPlanNodeExecution` work once Phase 1 is implemented, or is Change
  B (`define-provider-prepared-kernel-execution-contract`) actually needed?~~
  **Resolved during Phase 1 (task 13.1-13.3).** The original `submit`/
  `complete` pair, shaped around `ComputeExecutionPlan`, does not cleanly
  carry `KernelInvocation`/`KernelResult` work -- confirmed empirically:
  reusing it would have meant serializing Kernel-level data into a
  `ComputeExecutionPlan`-shaped `ProviderExecutionRequest`, which none of the
  fields fit. But the fix was to add `submit_kernel`/`complete_kernel` (plus
  `read_tensor`/`write_tensor`/`allocate_workspace`) as new, optional,
  defaulted methods on the *same* `ProviderExecutionApi` trait, not to define
  a wholly new contract. Since this stayed an ordinary implementation change
  to an existing trait in this crate -- not a new semantic decision needing
  separate governance -- `define-provider-prepared-kernel-execution-contract`
  is not needed. Task group 14 (Reference CPU extraction) can proceed
  directly against the extended `ProviderExecutionApi`.
- **A different question than the one above turned out to need Change B
  after all.** The submit/complete payload-shape question above was
  resolved without it; separately, task groups 3 (task 3.3) and 5 (tasks
  5.4-5.6) later found that `read_tensor`/`write_tensor`/`write_tensor_admitted`
  being `HostTensor`-typed *on the trait itself* -- not merely "does
  submit/complete carry Kernel work" -- blocks a genuinely device-resident
  Provider from implementing `ProviderExecutionApi` at all, which
  `device-resident-resource`'s existing spec already requires be possible.
  That is a new semantic decision (what does a Provider-agnostic tensor
  *value* look like), not an implementation catching up to an
  already-correct spec, so `define-provider-prepared-kernel-execution-contract`
  was opened as its own Change and landed a `TensorValue` contract
  (additive, alongside the existing `HostTensor`-typed methods). See tasks
  3.3 and 5.6 in `tasks.md` for exactly what that Change did and did not
  close here.
- Should `externalize-runtime-extension-modules` (Change C) become a
  normative OpenSpec statement that Components/Formats/Providers live outside
  the Core repository, or stay a repository/packaging decision? Not decided
  here; revisit after Phase 7.
