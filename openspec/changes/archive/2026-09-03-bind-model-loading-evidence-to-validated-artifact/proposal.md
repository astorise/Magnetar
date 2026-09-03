## Why

A sixth audit round of PR #36 (HEAD `9939232`), a full re-audit rather than a
narrow revalidation, confirmed every round-5 closure held (resume
revalidation, mandatory inventory completeness, residency-without-Provider-
write rejection) and found a deeper P0 the prior five rounds' narrower fixes
did not reach: **evidence of successful Model Loading and successful weight
materialization is represented by caller-constructible state instead of
Runtime-issued authority.** Concretely, verified directly against the code
before accepting the audit's framing:

- `LoadedModelContext` and its nested `ModelLoadingResidencyPlan` have every
  field `pub`, with no crate-internal constructor -- an external caller
  could hand-construct one claiming any `ModelArtifactId`/`ModelLoadingState`
  without `ModelLoadingCoordinator::load()` (and the trust/digest validation
  it performs) ever having run. `Runtime::create_model_instance()` accepts
  `&LoadedModelContext` and trusts it outright.
- `ModelInstance.definition.resource_bindings` (and its `weights`/
  `memory_allocations` fields) remain fully `pub`, reachable from any caller
  holding `Runtime::model_instances_mut()`. Round 5's `weights_materialized`
  check (`derive_effective_readiness_checks` calling
  `ProviderExecutionApi::read_tensor`) proves *some* Provider currently holds
  bytes under each required `TensorResourceId` -- it does not prove those
  bytes came from this instance's own materialization transaction, so a
  caller retaining ordinary access to `write_tensor`/`record_tensor_residency`
  /`resource_bindings.weights` can still assemble a passing state by hand.

Both gaps share one root cause and are fixed together here. A companion P1
(the same audit round, and this Change's design) is closed as a direct
consequence: readiness derivation depends today on
`ProviderExecutionApi::read_tensor() -> Option<HostTensor>`, an API the trait
itself documents as host-CPU-shaped and provisional -- a device-only Provider
that never implements host readback could otherwise fail warmup solely for
respecting its own resident-storage model.

Scope was clarified directly with the auditor before starting: rather than
implementing inline, this was split into its own OpenSpec Change given its
size relative to every prior round (`define-model-component-graph-
production-contract`-scale, not `invalidate-tensor-residency-on-release`-
scale), so the design gets reviewed before the ~equivalent-of-round-2 code
churn happens.

A companion regression found by the same audit round -- the `9939232`
archive merge had overwritten, not merged, five anti-forgery scenarios and
three normative paragraphs already accepted into the canonical
`openspec/specs/model-instance/spec.md` -- was fixed directly (a spec-text
restore, not a design question) ahead of this proposal and is not part of
this Change's scope.

## What Changes

- `LoadedModelContext` and `ModelLoadingResidencyPlan` fields become
  `pub(crate)`; the only way to obtain one is
  `ModelLoadingCoordinator::load()` succeeding. **BREAKING** for any external
  crate that hand-constructs either type directly (grep confirms none of
  this workspace's own code does -- `contract_tests` already goes through
  `load()`).
- `ModelInstanceDefinition.resource_bindings` and
  `ModelInstanceResourceBindings`'s `weights`/`memory_allocations` fields
  become `pub(crate)`, closing both direct-field mutation and
  wholesale-field-replacement (e.g. cloning one instance's bindings onto
  another). **BREAKING** for `contract_tests`' `bind_fake_weight` helper,
  which currently pokes these fields directly -- migrated onto the new
  public materialization entrypoint below, the same contract a real
  embedder must use.
- A new public `Runtime` entrypoint (promoting the existing private,
  already-correct `WeightMaterializationTransaction`-backed
  `materialize_model_instance_weights` production path) becomes the *one*
  legitimate way any caller -- production code or an external embedder --
  turns named weight bytes into bound, Ready-eligible resources for a Model
  Instance.
- That transaction's commit step now mints a Runtime-owned
  `MaterializationEvidence` record per instance (artifact id + the exact
  committed resource-id set), replacing an unload/rollback-cleared slot each
  time. `derive_effective_readiness_checks`'s `weights_materialized` now
  requires this evidence to exist, match the instance's own declared
  `artifact` id, and match the currently-bound resource-id set -- instead of
  probing Provider storage via `read_tensor`.
- Non-goal (explicit, see design.md): verifying that materialized bytes are
  bit-identical to the specific validated Model Artifact's declared tensor
  content. Evidence proves *this instance's own transaction* produced the
  binding (closing "some other caller wrote these bytes by hand" and
  "instance B reused instance A's evidence"); it does not add a general
  per-tensor content digest check beyond what already exists as one
  fixture-specific check outside this layer. Left to a named follow-up.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `model-loading`: `LoadedModelContext`/`ModelLoadingResidencyPlan` gain an
  explicit requirement that they are Runtime-issued and not externally
  constructible.
- `model-instance`: "Model Instance Readiness" gains a requirement that
  `weights_materialized` is derived from Runtime-issued, artifact-bound,
  instance-bound materialization evidence rather than a Provider-storage
  presence probe; "Model Instance Creation" (or a new requirement in the
  same spec) gains that weight resource bindings are not externally
  settable outside the one authorized materialization transaction.

## Impact

- `magnetar-runtime/src/model_loading.rs`: field visibility on
  `LoadedModelContext`/`ModelLoadingResidencyPlan`.
- `magnetar-runtime/src/model_instance.rs`: field visibility on
  `ModelInstanceDefinition.resource_bindings` and
  `ModelInstanceResourceBindings`; new `MaterializationEvidence` storage.
- `magnetar-runtime/src/first_native_runtime.rs`: `materialize_model_instance_
  weights` (or a thin public wrapper) becomes the public entrypoint;
  `WeightMaterializationTransaction::commit` mints evidence.
- `magnetar-runtime/src/inference_api.rs`: `derive_effective_readiness_checks`
  switches `weights_materialized` from `read_tensor` probing to evidence
  matching.
- `magnetar-runtime/tests/contract_tests/model_instance.rs`: `bind_fake_weight`
  migrated onto the new public entrypoint.
- `magnetar-runtime/src/tests.rs` / `first_native_runtime/tests.rs`: new
  tests for forged-context rejection, hand-written-weights rejection,
  cross-instance/cross-artifact evidence reuse rejection, and the
  device-only-Provider-no-readback happy path.
