# Changelog

## v0.1.0 - Unreleased

### Added

- `magnetar-runtime` local inference runtime baseline.
- `magnetar-cli` boundary harness for local run, chat, model, provider,
  device, and serve workflows.
- Runtime contracts for Component, Capability, Provider, Device, Resource
  Affinity, Resolution Policy, Memory, Tensor, Operator, Kernel Registry,
  Kernel Dispatch, Reference CPU, Model Artifact, Model Loading, Model
  Instance, Tokenizer, Generation, Sampling, Session, KV Cache, Prefix Cache,
  Continuous Batching, Runtime Inference API, and E2E conformance.
- Release packaging, release security, release conformance, and v0.1 cutover
  policy modules with executable validation coverage.

### Changed

- Component and Model artifact trust decisions now fail closed for
  self-asserted publisher/source identity. Digest pinning and explicit local
  development policy remain the accepted non-signature trust mechanisms.
- README implementation status now separates implemented contracts, fixture
  paths, preview behavior, deferred roadmap items, and unsupported v0.1 scope.
- Canonical OpenSpec specs now include release-relevant purpose statements
  instead of archive-generated placeholders.
- First-native Qwen prefill/decode now executes through `ExecutionGraph` plus
  published `PreparedExecutionPlan` bindings, with no hot-path Kernel Registry
  rediscovery once a plan generation is ready.
- First-native compute now resolves its Provider and allocates through the
  Runtime-owned `MemoryManager` exclusively; the production compute path no
  longer constructs a local `ReferenceCpuExecutor` or `MemoryManager` of its
  own.
- Executed model weights are now bound to Runtime resources created by model
  loading and referenced by the active `ModelInstance`, not a fixture-weight
  side channel.
- KV cache data is now a Runtime-owned resource with session/model
  instance/layer/affinity/accounting metadata and transactional prepare,
  commit, and abort semantics tied to sampling and token-commit success.
- Decode now passes the correct absolute position for the token being
  decoded, covered by a multi-step generation-loop oracle.
- Per-step causal observations (Model Instance readiness re-checked every
  step, graph validation) replace evidence booleans that were previously
  fixed at the call site; the Runtime Inference API observation buffer is
  now capacity-bounded rather than unbounded.
- `magnetar chat` now executes every turn of a chat session through one
  persistent Runtime `Runtime` and `InferenceSessionId`; cancellation and
  close act on that same session rather than a session no turn used.
- Production coverage measurement now excludes `first_native_runtime`'s unit
  tests (moved to a sibling `tests.rs` file, matching every other module's
  convention) instead of counting embedded `#[cfg(test)]` source as Runtime
  implementation source.

### Fixed

- Prevented forged Component publisher/source metadata from granting trust.
- Prevented forged Model provenance publisher metadata from granting trust.

### Known Limitations

- The E2E success path still uses a tiny synthetic fixture architecture and
  fixture tokenizer for its numerics, not a published Qwen checkpoint or
  production tokenizer artifact.
- Production model hub downloads, production server API, GPU Providers,
  production CLI UX, and agent/tool Runtime execution are outside v0.1 scope.
- Architecture Freeze #1 is **accepted** at commit `f71b346`
  (2026-09-03, CI run
  https://github.com/astorise/Magnetar/actions/runs/33746039192, all 10
  jobs green). An external audit of commit `0197be1` (PR #36) had
  correctly found two P0 gaps in the weight-materialization lifecycle
  that an earlier acceptance declared at that commit missed: (1)
  `ModelInstance` was marked `Ready` immediately on creation, before
  mandatory weights were ever bound, and `acquire_usage`'s own readiness
  check did not inspect weight bindings -- only a deeper, incidental
  graph-dispatch check happened to catch a missing weight, which is not
  the same as the instance's own reported readiness being trustworthy;
  (2) weight materialization was not transactional:
  `Provider.write_tensor` was called before `MemoryManager` admission
  (inverting the intended authority order), a residency registration
  error was silently discarded (`let _ = record_tensor_residency(...)`),
  a mid-loop failure left already-written weights unrolled-back, and
  `unload_model_instance` released `MemoryAllocationId`s but never
  called `Provider.release_tensor` for weight `TensorResourceId`s,
  leaking Provider-owned storage across repeated load/unload cycles.
  Both are now fixed by `openspec/changes/transactional-weight-
  materialization`: creation no longer auto-readies (the caller must
  explicitly reach `Ready` through the new `ModelInstances::mark_ready`,
  used only once every weight is bound); a new
  `WeightMaterializationTransaction` mirrors the KV cache path's
  already-correct admission-first, error-propagating,
  abort-releases-everything pattern; and `unload_model_instance` now
  resolves each released weight's owning Provider generically (via
  `TensorResidency`/`ResourceAffinity`, no hardcoded Provider name in
  the generic Runtime layer) and releases its storage. Verified: full
  workspace test suite (1,393 tests across the lib, CLI, and
  `contract_tests` integration binaries) passing, coverage ratchet at
  78.89% matching baseline, `openspec validate --all --strict` 79/79,
  live `magnetar run qwen-test "Hello"` unaffected, and the CI run cited
  above.
- Release artifacts are not final until generated from the exact release commit
  and tag.

### Security Notes

- Cryptographic artifact signing is not implemented for v0.1 and must not be
  claimed as present; it is deferred to a dedicated future change and this
  release makes no authenticated publisher identity claim.
- Cache presence, model alias, local file location, recognized format,
  publisher metadata, source metadata, and fixture markers do not grant trust by
  themselves.
