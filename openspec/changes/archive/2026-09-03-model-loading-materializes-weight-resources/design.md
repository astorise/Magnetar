## Context

`reach-architecture-freeze-1`'s Group 8 investigation read `ModelLoadingCoordinator::load()`
end to end and found it does exactly two things toward "materialization":
admit one aggregate `MemoryAllocationRequest` sized to
`plan.expected_resident_bytes`, and emit two observation events with no
tensor bytes touched. The actual per-tensor work -- writing each weight into
Provider storage, admitting it individually, and recording
`ModelInstance.resource_bindings.weights` -- already exists, correctly, as
`materialize_model_instance_weights` in `first_native_runtime.rs`, called
from `load_fixture_instance` right after `create_model_instance`. It has
zero Qwen/fixture dependency in its own logic (only its caller,
`bind_qwen_fixture_weights`, is fixture-specific -- the digest check).

What is missing is not the logic. It is: (a) this phase being recognized by
the `model-loading`/`model-instance` specs as part of the Model Loading
contract at all, and (b) `ModelInstanceReadinessChecks` having any way to
know whether it happened. `ModelInstanceReadinessChecks` already has the
exact extensibility point this needs --
`kernel_preparation_ready`/`autotuning_ready` are both booleans defaulting
`true`, set explicitly by whichever caller's flow actually performs that
check, folded into `readiness()`'s existing Failed/Suspended/Ready logic.

This does not require a real Model Artifact byte format. `formats/gguf`/
`formats/safetensors` are still empty submodule templates (separate repos,
out of reach this session); the E2E fixture's in-memory `BTreeMap<String,
HostTensor>` is a sufficient stand-in weight source for building and
testing this contract, exactly as Reference CPU stands in for a real
Provider elsewhere.

## Goals / Non-Goals

**Goals:**
- Recognize weight materialization as a named, generic phase of the Model
  Loading lifecycle, callable by any weight source (fixture today, a real
  format loader later), not fixture-specific code.
- Make `ModelInstanceReadinessChecks` aware of whether it happened and
  succeeded, so a missing/failed materialization is a structured readiness
  failure, not a deep, opaque dispatch-time error.
- Preserve the existing three-step sequence (`load()` -> `create_model_instance`
  -> materialize weights) rather than inventing a new one -- it is already
  the right shape, per `load()`'s Lazy Loading Policy constraint (below).

**Non-Goals:**
- Building a real GGUF/Safetensors parser or wiring `formats/gguf`/
  `formats/safetensors`. Out of reach this session (separate, currently-empty
  submodule repos); this Change defines the contract a real loader would
  call, the same way `define-provider-prepared-kernel-execution-contract`
  defined a contract without building a real GPU Provider.
- Changing `load()`'s own signature to accept a Provider or weight bytes
  directly. See Decision 1.
- Changing how `MemoryManager` accounts the *aggregate* residency
  allocation `load()` already performs -- that stays; per-tensor admission
  is additional accounting alongside it, exactly as
  `materialize_model_instance_weights` already does today.

## Decisions

### Decision 1: Materialization stays a distinct step after `load()`, not folded into it

**Choice:** Keep the existing sequence: `load()` (aggregate residency
allocation only) -> `create_model_instance()` -> an explicit
materialization call. Formalize the third step's function signature and
move it to `model_loading.rs`; do not add a Provider parameter to `load()`
itself.

**Rationale:** `model-loading`'s existing "Lazy Loading Policy" requirement
means `load()` must be callable without weight bytes being ready yet --
folding materialization into `load()`'s own signature would force every
caller (including genuinely lazy ones) to supply a Provider and weight
source it may not have at load time. The current three-step shape already
accommodates this correctly; the problem was never the sequencing, only
that step three is unofficial. This also matches
`materialize_model_instance_weights`'s existing dependency on `instance:
&ModelInstanceId`, which does not exist until after `create_model_instance`
runs -- materialization structurally cannot move earlier than that without
also restructuring `ModelInstance` creation itself, which is out of scope.

**Alternative considered:** Add materialization as a phase inside `load()`
itself, gated by an `Option<(Arc<dyn ProviderExecutionApi>, WeightSource)>`
parameter. Rejected: this either breaks the "no `ModelInstanceId` yet"
ordering constraint (materialization would have to defer its own
`ModelInstance.resource_bindings` write until instance creation happens
anyway, achieving nothing by moving into `load()`) or requires restructuring
instance creation to happen inside `load()`, a much larger change than this
Change's actual gap warrants.

### Decision 2: `weights_materialized` as a plain readiness boolean, following the existing pattern exactly

**Choice:** Add `weights_materialized: bool` (default `true`) to
`ModelInstanceReadinessChecks`, folded into `readiness()` the same way
`kernel_preparation_ready`/`autotuning_ready` already are (any of these
being `false` => `Failed`). A caller that never materializes weights (or
whose Model Instance genuinely has none) leaves the default `true` and is
unaffected; a caller that does materialize sets it from the real outcome.

**Alternatives considered:**
- A richer `WeightMaterializationState` enum (`NotStarted`/`InProgress`/
  `Complete`/`Failed`) instead of a bool: rejected as more than this gap
  needs -- none of the other seven existing checks in
  `ModelInstanceReadinessChecks` carry richer state than a bool either
  (`residency_available`, `provider_ready`, etc. are all plain booleans),
  and introducing one just for this check would be inconsistent with the
  established pattern for no added benefit `readiness()`'s binary
  Failed/Suspended/Ready output could use.
- Deriving `weights_materialized` implicitly from whether
  `resource_bindings.weights` is non-empty: rejected -- indistinguishable
  from "this Model Instance genuinely has zero declared weights" (a
  degenerate but not inherently invalid case), and does not capture
  "materialization was attempted and failed partway" distinctly from
  "materialization was never attempted."

## Risks / Trade-offs

- [Existing callers of `create_model_instance` that do not yet call the
  relocated materialization function would see no behavior change (default
  `true`), silently continuing to skip the new readiness check] → Intended,
  not a gap: matches how `autotuning_ready`/`kernel_preparation_ready`
  already work, and existing production code (`load_fixture_instance`)
  already does materialize weights immediately after instance creation, so
  it gains the real check rather than silently defaulting past it.
- [Moving `materialize_model_instance_weights` out of `first_native_runtime.rs`
  touches its one real caller (`bind_qwen_fixture_weights`) and every test
  that exercises weight binding indirectly] → Mitigated by keeping the
  function's signature and behavior identical, only its location and name
  changing; full first-native test suite re-run after the move confirms no
  behavioral regression.
