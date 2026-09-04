## Why

A tenth audit round of PR #36 found that the ninth round's fixes closed
*direct construction* of forged authority (`ModelTrustDecision::new`,
`ModelInstanceDefinition.artifact`/`.architecture`) but left the *caller
still able to reach the same outcome one level up*, through paths that
remain fully `pub`:

- `ModelTrustStore` (`model.rs`) has every field `pub` and a fully public
  builder API. Any caller can construct their own store, self-declare a
  digest trusted via `.trust_digest(...)`, and obtain a `Trusted` decision
  for whatever artifact they choose before calling `load_model` -- sealing
  `ModelTrustDecision::new` closed fabricating a decision out of nothing,
  not this.
- `Runtime::create_model_instance(loaded, architecture, affinity)` accepts
  `architecture`/`affinity` as caller-supplied parameters never compared
  against `loaded.plan()`'s own Runtime-resolved values, even though the
  plan already carries a resolved `ModelArchitecture` and, when the loading
  phase settled on one, a `provider_binding`/`device_binding`.
- `materialize_model_instance_weights` accepts caller-constructed `HostTensor`s
  with no check against the manifest's declared `shape`/`storage_dtype` for
  that tensor name. A tensor the artifact actually declares as quantized has
  no digest (digests are F32-only, per `bind-materialized-weight-content-to-
  model-artifact-digests`'s documented limitation), so a caller can supply a
  fabricated F32 `HostTensor` under its name and nothing rejects it -- silently
  bypassing the format parser's correct refusal to materialize non-F32 content.

All three share one root cause: Runtime has no enforced boundary between
values a caller *asserts* and values Runtime itself already *resolved*
during loading/planning. Investigation before writing this proposal found
that closing the first of these for real is not a narrow field seal --
`load_model`/`load_model_observed` (`inference_api.rs`) are free functions
over a caller-owned `ModelLoadingCoordinator`/`MemoryManager`; `Runtime`
holds neither today, confirmed by grep across every real call site (CLI,
the qwen-test live E2E fixture, and both test modules). Sealing trust
requires actually coupling model loading to a `Runtime`-owned, once-configured
trust policy, not just hiding fields on `ModelTrustStore`.

## What Changes

- **BREAKING**: `load_model`/`load_model_observed` (`inference_api.rs`) stop
  taking a caller-supplied `trust: &ModelTrustDecision`. `RuntimeBuilder`
  gains a `.trust_store(ModelTrustStore)` configuration method (default:
  `ModelTrustStore::default()`, i.e. deny-all -- identical to today's CLI
  behavior), `Runtime` retains it sealed (no public getter that returns an
  owned, re-mutable store), and evaluates trust internally from the
  manifest. Every real call site (CLI's `run_load_model`, the qwen-test
  live E2E fixture, both test modules, and any `contract_tests` site) is
  migrated to build/configure a `Runtime` up front instead of fabricating
  a `ModelTrustDecision` inline.
- **BREAKING**: `Runtime::create_model_instance` rejects `architecture`/
  `affinity` that disagree with `loaded.plan()`'s already-resolved values:
  `architecture.architecture` must equal `plan().architecture`; when the
  plan resolved a `provider_binding`/`device_binding`, `affinity`'s
  provider/device must agree (an unresolved plan field imposes no
  constraint, consistent with this crate's existing `None`-is-permissive
  precedent). `kind`/`required_capabilities` on `ModelArchitectureImplementation`
  have no plan counterpart and remain legitimate caller choices. A new
  `ModelInstanceError` variant reports the mismatch.
- `LoadedModelContext`/`ModelInstanceDefinition` gain a
  `required_weight_shapes` map (`BTreeMap<String, (Vec<u64>, ModelDType)>`),
  threaded the same way `required_weight_digests` already is.
  `WeightMaterializationTransaction::stage_weight` rejects a staged tensor
  whose shape disagrees with the declared shape, or whose declared
  `storage_dtype` is not `F32` at all (regardless of whether a digest
  exists), before the existing digest check runs. A new `InferenceApiError`
  variant reports the mismatch.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `model`: `ModelTrustStore` gains a requirement that trust evaluation for
  a `load_model` call is Runtime-configured authority, not caller-supplied
  per call.
- `inference-api`: "Model Loading API" gains a requirement that trust is
  sourced from the `Runtime` performing the load, not a caller-supplied
  `ModelTrustDecision` parameter.
- `runtime`: gains a requirement that `RuntimeBuilder` is the sole
  configuration point for trust policy, set once and sealed for the
  `Runtime`'s lifetime.
- `model-instance`: "Model Instance References Architecture Implementation"
  gains a requirement that caller-supplied architecture/affinity must agree
  with the loading plan's already-resolved values; "Materialized Weight
  Content Matches Its Declared Digest When One Exists" is joined by a
  sibling requirement that materialized content must also match the
  artifact's declared shape and storage dtype, independent of digest
  presence.

## Impact

- `magnetar-runtime/src/model.rs`: no field-visibility change to
  `ModelTrustStore` itself (it stays a legitimate value type for
  deployment-time configuration) -- the seal is that `load_model` stops
  accepting a caller-supplied decision, not that the store becomes
  unconstructible.
- `magnetar-runtime/src/runtime.rs`: `RuntimeBuilder` gains `.trust_store(..)`;
  `Runtime` gains a sealed trust field and an internal evaluation path;
  `create_model_instance` gains the plan cross-check.
- `magnetar-runtime/src/inference_api.rs`: `load_model`/`load_model_observed`
  signature change (drop `trust`, take the owning `Runtime`); new
  `InferenceApiError` variant for the weight shape/dtype mismatch.
- `magnetar-runtime/src/model_loading.rs`: `LoadedModelContext` new
  `required_weight_shapes` field, populated in
  `ModelLoadingCoordinator::load()`.
- `magnetar-runtime/src/model_instance.rs`: `ModelInstanceDefinition` new
  `required_weight_shapes` field threaded through `from_loaded_context`;
  new `ModelInstanceError` variant for the architecture/affinity mismatch.
- `magnetar-runtime/src/first_native_runtime.rs`:
  `WeightMaterializationTransaction::stage_weight` shape/dtype check;
  every call site there that currently fabricates a `ModelTrustDecision`
  inline migrates to a configured `Runtime`.
- `magnetar-cli/src/commands.rs`: `run_load_model` migrates from a bare
  `ModelLoadingCoordinator`/`MemoryManager`/inline trust to a configured
  `Runtime`.
- `magnetar-runtime/src/tests.rs`, `magnetar-runtime/tests/contract_tests/`:
  every `load_model`/`load_model_observed` call site migrated; new tests
  for architecture mismatch, affinity mismatch, weight shape mismatch, and
  non-F32-declared-dtype rejection.
