## Why

A sixth audit round of PR #36 identified, as one half of a single P0
("provenance"), that `materialize_model_instance_weights`'s public
`&BTreeMap<String, HostTensor>` accepts caller-supplied tensor bytes with
no verification against the validated Model Artifact's declared content --
a caller can supply arbitrary bytes under the correct tensor names and the
Runtime accepts them as if they were the artifact's real weights.
`bind-model-loading-evidence-to-validated-artifact` (the Change that fixed
the other half -- caller-constructible `LoadedModelContext`/materialization
evidence, and cross-instance evidence reuse) scoped byte-content
provenance out as an explicit Non-Goal, reasoning that the round-6 audit's
P0 was "specifically about caller-constructible evidence." A seventh
audit round reviewed and accepted that scoping (downgrading the residual
to P1); an eighth round re-read round-6's actual text and found this
characterization inaccurate -- round-6 explicitly treated both halves as
one architectural P0, recommending they be fixed together. That
correction is recorded directly in `CHANGELOG.md`'s history rather than
silently reconciled.

This Change closes the second half for real: materialized weight bytes
become verifiable against a Runtime-observable, artifact-declared content
digest before the transaction that stages them succeeds, for any manifest
that declares one -- generic per-tensor content binding, not a
fixture-specific special case.

## What Changes

- `ModelTensorMetadata` (`model.rs`) gains an optional per-tensor content
  digest field, mirroring the existing whole-artifact (`ModelArtifactId.
  digest`), part (`ModelArtifactPart.digest`), and shard (`ModelShard.
  digest`) digest fields this crate already has -- just at tensor
  granularity, which none of those cover today.
- The E2E fixture's tensor inventory (`e2e_fixture_weight_inventory` and
  its manifest YAML, `first_native_runtime.rs`) is extended to populate
  real per-tensor digests computed from the actual checked-in
  `.safetensors` file's bytes -- the one real artifact source that exists
  in this crate today (`model_format_roadmap.rs`'s Safetensors/GGUF
  parsers remain unimplemented roadmap contracts, confirmed by direct
  investigation before starting this Change).
- `LoadedModelContext`/`ModelInstanceDefinition` gain a
  `required_weight_digests` field (same threading shape as
  `required_weight_names`: populated once at `ModelLoadingCoordinator::
  load()` time from `manifest.tensors`, `pub(crate)`, empty means
  "unknown" -- consistent with the existing precedent for how this crate
  threads manifest-declared facts into readiness-relevant state without
  widening `create_model_instance`'s or `from_loaded_context`'s public
  signatures).
- `WeightMaterializationTransaction::stage_weight`
  (`first_native_runtime.rs`) verifies each staged tensor's bytes against
  its declared digest (when one exists for that tensor name) before
  admission/write succeed; a mismatch aborts the transaction with a
  dedicated error, the same class of failure as a memory-admission or
  residency-registration failure today.
- No change to `materialize_model_instance_weights`'s or
  `create_model_instance`'s public signatures.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `model-artifact`: `ModelTensorMetadata` gains an explicit per-tensor
  content digest requirement, and a requirement that a declared digest
  constrains what content counts as that tensor for the artifact it
  belongs to.
- `model-loading`: `LoadedModelContext` gains a requirement that
  manifest-declared per-tensor digests are carried through to Model
  Instance readiness derivation.
- `model-instance`: "Model Instance Readiness" (or a new requirement in
  the same spec) gains a requirement that materialized weight content
  matching its declared digest is a precondition for that tensor counting
  as legitimately materialized, when the artifact declared one.

## Impact

- `magnetar-runtime/src/model.rs`: `ModelTensorMetadata` new field; its
  YAML deserialization counterpart struct and conversion logic.
- `magnetar-runtime/src/model_loading.rs`: `LoadedModelContext` new field,
  populated in `ModelLoadingCoordinator::load()`.
- `magnetar-runtime/src/model_instance.rs`: `ModelInstanceDefinition` new
  field, threaded through `from_loaded_context`.
- `magnetar-runtime/src/first_native_runtime.rs`:
  `WeightMaterializationTransaction::stage_weight` verification;
  `e2e_fixture_weight_inventory`/`e2e_fixture_manifest` populate real
  per-tensor digests from the checked-in Safetensors file.
- `magnetar-runtime/src/tests.rs` / `first_native_runtime/tests.rs`: new
  tests for content-mismatch rejection, matching-content acceptance, and
  the "no digest declared" permissive-fallback case.
