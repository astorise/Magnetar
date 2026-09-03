## Why

`ModelLoadingCoordinator::load()` (`model_loading.rs:718`) admits one
*aggregate* `MemoryAllocationRequest` sized to the whole model, then
"materialization" is exactly two observation events
(`MaterializationStarted`/`MaterializationCompleted`) -- no tensor bytes
read, no `TensorResourceId` created, no Provider interaction, for any
artifact type. `reach-architecture-freeze-1`'s investigation (task group 8)
found the actual per-tensor work already exists, just not inside `load()`
or anywhere the `model-loading`/`model-instance` specs recognize: a bolt-on
function (`materialize_model_instance_weights`, `first_native_runtime.rs`)
called *after* `create_model_instance`, from Qwen-fixture-specific code,
writes each weight into Provider storage, admits it through the Runtime
`MemoryManager`, and populates `ModelInstance.resource_bindings.weights` --
correctly, but as an unofficial afterthought rather than a phase of the
Model Loading contract.

Two concrete problems follow from that being unofficial:

1. Nothing in `ModelInstanceReadinessChecks` (`model_instance.rs`) knows
   this step exists. An instance can be marked `Ready` -- and pass every
   check `Generation Requires Ready Model Instance` asks for -- whether or
   not any weight was ever actually bound. A missing or failed
   materialization step surfaces only much later, as an opaque "no
   materialized data for input resource" error deep inside first Kernel
   dispatch, not as a structured readiness failure at the boundary that
   already exists for exactly this purpose (`provider_ready`,
   `kernel_preparation_ready`, `autotuning_ready`, ... all follow this
   pattern already).
2. `materialize_model_instance_weights` is named and positioned as if it
   were still fixture-specific support code, even though its logic already
   has zero Qwen/fixture dependency (task 8.6 in `reach-architecture-freeze-1`
   already reduced `bind_qwen_fixture_weights` to a thin digest-check
   wrapper around it). A future real Model Artifact loader (GGUF,
   Safetensors -- not yet implemented; `formats/gguf`/`formats/safetensors`
   are still empty submodule templates, out of scope here) has no
   recognized, spec-governed entry point to call instead of reinventing this
   logic itself.

This Change does not require a real byte-level Model Artifact format to
exist: the gap is in the Runtime-side *contract* (what Model Loading's
lifecycle recognizes as a phase, and what Model Instance readiness checks),
not in parsing real files. The existing in-memory `BTreeMap<String,
HostTensor>` weight source (today: the E2E fixture's; tomorrow: whatever a
real format parser produces) is sufficient to build and test this contract,
the same way Reference CPU stands in for a real Provider elsewhere in this
codebase.

## What Changes

- Formally recognize weight materialization as a named, generic phase of
  the Model Loading lifecycle (spec-level requirement below), documented as
  such at its existing location -- **not** physically relocated into
  `model_loading.rs` after all: that module deliberately has no dependency
  on the full `Runtime` type (`load()` itself takes only `&mut MemoryManager`,
  never `&mut Runtime`; `runtime.rs` depends on `model_loading.rs`, not the
  reverse), and this function needs `&mut Runtime` (Provider resolution,
  `ModelInstance` mutation, lifecycle transition on failure). Moving it
  would introduce a mutual file dependency for no behavioral benefit --
  found while implementing, not anticipated in the original design.md, and
  corrected here rather than forced through anyway.
- Add a `weights_materialized` readiness check to
  `ModelInstanceReadinessChecks` (`model_instance.rs`), following the exact
  existing pattern `kernel_preparation_ready`/`autotuning_ready` already
  use. **This alone does not protect the actual first-native production
  path** -- found during implementation: `ModelInstances::create()` calls
  `mark_ready()` unconditionally right after creation, via a *separate*,
  simpler lifecycle-transition-based readiness mechanism
  (`readiness_for_lifecycle`) that never consults
  `ModelInstanceReadinessChecks` at all. That struct is real and used by a
  different consumer (`warm_model_instance`); it is additive value for that
  path, not the fix for this one.
- The actual protective fix: when weight materialization fails, explicitly
  transition the Model Instance's lifecycle away from `Ready` (to `Failed`,
  an already-legal transition) before returning the error, so a Runtime
  that already inserted the instance as `Ready` (per `create()`'s existing,
  unconditional behavior) does not keep claiming so after materialization
  is known to have failed -- rather than relying on every caller to notice
  and react to the propagated `Result::Err` itself.
- No change to `load()`'s own signature or the Lazy Loading Policy
  requirement: materialization stays a distinct, explicit step (as it
  already effectively is today), not folded into `load()` itself --
  `load()` must remain callable without immediately requiring weight bytes
  to exist, which is exactly what lazy loading needs.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `model-loading`: add a requirement that Model Loading's lifecycle
  recognizes an explicit, generic weight-materialization phase producing
  per-tensor `TensorResourceId`s through a Provider, distinct from the
  aggregate residency allocation `load()` already performs.
- `model-instance`: extend "Model Instance Readiness" to also consider
  weight materialization state, alongside the residency/Provider/Device/
  adapter/memory-pressure/policy/architecture-readiness factors it already
  names.

## Impact

- `magnetar-runtime/src/model_loading.rs`: new generic weight-materialization
  function (moved/adapted from `first_native_runtime.rs`'s
  `materialize_model_instance_weights`). No new field on `LoadedModelContext`:
  the resulting `TensorResourceId` bindings already land directly on
  `ModelInstance.resource_bindings.weights`, which is created *after*
  `LoadedModelContext` is consumed -- there is nothing for `LoadedModelContext`
  itself to carry.
- `magnetar-runtime/src/model_instance.rs`: `ModelInstanceReadinessChecks`
  gains `weights_materialized: bool`; `readiness()`/`validate()` account for
  it the same way they already account for `kernel_preparation_ready`.
- `magnetar-runtime/src/first_native_runtime.rs`: `bind_qwen_fixture_weights`
  keeps only its fixture-specific digest check, now calling the
  relocated/generalized function instead of owning the logic itself; the
  call site that currently sets instance readiness needs to set
  `weights_materialized` from this step's real outcome.
- Test coverage: existing weight-binding tests
  (`e2e_weight_binding_rejects_tampered_artifact_bytes`,
  `e2e_graph_execution_fails_closed_on_missing_weight`,
  `e2e_unload_releases_weight_resource_allocations`,
  `e2e_weight_resources_are_isolated_per_model_instance`,
  `e2e_weight_byte_change_alters_generated_logits`) SHALL continue passing
  unchanged; new coverage for the readiness-check addition itself.
- Unblocks (in `reach-architecture-freeze-1`, tracked there once this
  Change lands): tasks 8.3, 8.4, 8.5 (already true in effect, now true
  *and* spec-recognized) and narrows what remains open in 8.1/8.2/8.6 to
  exactly the real-artifact-format question, which stays genuinely out of
  reach without the external submodule work.
