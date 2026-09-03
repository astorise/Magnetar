## Why

A fifth-round revalidation audit of PR #36 (commit `dd280df`) confirmed
`seal-model-instance-readiness-authority` (round 2) closed every gap it
targeted, then found two further real gaps in the same family, both
already contradicting pre-existing canonical spec text (spec-correct,
code non-compliant, not a new architectural decision):

1. `resume_model_instance` reached `Ready` via `ModelInstance::resume()`'s
   internal `Suspended -> Loading -> Ready` transitions without ever
   calling `derive_effective_readiness_checks` -- state that made an
   instance eligible for suspension (Provider health, weight residency,
   Device availability) could have changed while suspended, and resume
   would still jump straight back to `Ready` on stale assumptions.
2. `weights_materialized` derivation (round 2) checked that every *bound*
   weight had a real residency, but never checked (a) that the bound set
   was the *complete* mandatory set the loaded manifest declares (`model-
   loading`'s pre-existing "Qwen Loading Validates Tensor Inventory" and
   "Partial Loading Policy" requirements), or (b) that a residency's
   claimed Provider had actually received a `write_tensor` call for it --
   a caller could record a fully legitimate-looking `TensorResidency`
   (real Memory Manager allocation, real `record_tensor_residency` call)
   for a resource no Provider ever wrote.

## What Changes

- `ModelInstance::resume()` now only transitions `Suspended -> Loading`;
  `resume_model_instance` completes the transition through
  `warm_model_instance`, re-deriving readiness against current state.
- `LoadedModelContext` gains `required_weight_names: BTreeSet<String>`,
  populated by `ModelLoadingCoordinator::load()` from the manifest's
  declared tensor names. `ModelInstanceDefinition` gains a `pub(crate)`
  field of the same name, carried through `from_loaded_context`.
- `derive_effective_readiness_checks` now also requires (a) every name in
  a non-empty `required_weight_names` to be present as a bound key, and
  (b) for every bound weight, that the Provider its residency names
  actually holds the tensor (`ProviderExecutionApi::read_tensor`), not
  just that a residency record exists.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `model-instance`: "Model Instance Readiness" gains requirements that
  weight materialization proof requires both inventory completeness
  (when known) and real Provider-backed evidence per bound weight; a new
  "Model Instance Resume Revalidates Readiness" requirement is added.
  `model-loading`'s existing "Qwen Loading Validates Tensor Inventory"
  and "Partial Loading Policy" requirements already cover the inventory
  side of this Change (spec-correct, code non-compliant) and need no
  wording change -- `LoadedModelContext::required_weight_names` is the
  implementation catching up to spec text that already existed.

## Impact

- `magnetar-runtime/src/model_loading.rs`: `LoadedModelContext` field.
- `magnetar-runtime/src/model_instance.rs`: `ModelInstanceDefinition`
  field, `ModelInstance::resume` behavior change.
- `magnetar-runtime/src/inference_api.rs`: `resume_model_instance`,
  strengthened `derive_effective_readiness_checks`.
- Test coverage: 3 new dedicated tests (resume revalidation, incomplete
  inventory, residency-without-Provider-write); `contract_tests/model_instance.rs`'s
  `bind_fake_weight` helper updated to perform a real Provider write and
  bind under the fixture manifest's actual declared tensor name.
