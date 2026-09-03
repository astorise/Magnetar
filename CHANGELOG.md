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
- Architecture Freeze #1 is **accepted** at commit `e7dc45d`
  (2026-09-03, CI run
  https://github.com/astorise/Magnetar/actions/runs/33765211820, zero
  non-`success` jobs confirmed via `gh run view --json status,conclusion`
  and the jobs list directly -- that run's first pass had one failing
  job, `wasmtime component engine`, on a pre-existing test
  (`pushed_component_package_temp_materialization_is_removed_with_manager`)
  unrelated to this Change (a parallel-test race scanning the shared
  system temp directory for a filename prefix multiple tests use;
  confirmed by local reproduction under the exact CI feature flags,
  passing in isolation, and passing clean on this branch's prior 3 CI
  runs); a rerun of just that job came back green, so this is noted as
  a known pre-existing flake, not fixed as part of this Change). (The P0
  fix itself landed as `f71b346`; that push's own
  CI run failed on `clippy` for an unrelated reason -- a helper this
  same fix added, `check_repeated_load_unload_does_not_accumulate_
  weight_storage`, was missing the `#[cfg(test)]` attribute its sibling
  helpers all carry, so a fresh non-incremental build flagged it as
  dead code; my own local clippy pass had missed this because
  incremental compilation reused a stale artifact. Fixed by `ff06d98`,
  re-verified with `CARGO_INCREMENTAL=0` locally before repushing, and
  confirmed green by the CI run cited above -- this note exists so the
  accepted commit and the fix commit are not silently conflated.) An
  external audit of commit `0197be1` (PR #36) had
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
  the generic Runtime layer) and releases its storage. A follow-up
  revalidation audit of that fix (commit `633e942`) found one further
  gap: `MemoryManager::release(allocation)` only changes the
  `MemoryAllocation`'s own state and never removed the resource's
  `TensorResidency` record, so `tensor_residency()` kept reporting a
  weight as resident indefinitely after both its Provider storage and
  allocation were released -- a metadata leak growing on every failed
  materialization attempt and every load/unload cycle. Fixed by
  `openspec/changes/invalidate-tensor-residency-on-release`: a new
  `MemoryManager::remove_tensor_residency` is now called from both
  rollback and unload, in the order the audit itself recommended
  (release the Provider tensor, then remove the residency record, then
  release the Memory Manager allocation). A third audit round, this
  time a full-scope re-audit rather than a narrow revalidation
  (commit `a4d411b`), found readiness itself was still partly
  caller-declared: `warm_model_instance`'s public
  `ModelInstanceReadinessChecks` defaults `weights_materialized: true`,
  and the Runtime trusted that outright, so a caller using default
  checks against an instance whose weights were never materialized
  could warm it straight to `Ready`; `WarmupPolicy::Disabled`
  compounded this by publishing `readiness: Ready` without the
  lifecycle transition warmup's other policies perform, producing an
  internally inconsistent `lifecycle: Loading, readiness: Ready` state
  that `acquire_usage`/`generation_reference` (checking only readiness,
  never lifecycle) would still accept. **This means the "accepted"
  declaration above was live and technically inaccurate for the window
  between the previous audit round and this fix landing** -- noted here
  rather than silently corrected, per this document's own standard of
  not letting a known-false claim stand uncorrected. Scope was
  clarified directly with the auditor before fixing (their own initial
  reply mixed "necessary for this PR" with "desirable future
  architecture"): fixed by `openspec/changes/runtime-owned-model-
  instance-readiness-authority` without a warmup-API redesign --
  `warm_model_instance` now derives `weights_materialized`,
  `provider_ready`, and `device_ready` from actual Runtime state
  (resource bindings, the Provider registry, the Device list), ANDed
  with the caller's claim; `ModelInstance::validate_readiness` no
  longer lets a computed `Ready` readiness publish while the lifecycle
  hasn't itself reached a state that supports it; and
  `acquire_usage`/`generation_reference` now require both
  `lifecycle.supports_inference_use()` and
  `readiness.accepts_generation()` as a structural safety net.
  Deliberately not closed, and documented as an explicit trade-off
  rather than an oversight: `kernel_preparation_ready`/
  `autotuning_ready` remain caller-supplied (no generic Runtime-side
  derivation exists for these today), and the direct
  `ModelInstanceManager::mark_ready` bypass is not sealed by Rust
  visibility (would require editing 15 test call sites for a partial
  close, since `mark_ready` has no Runtime context to derive
  `provider_ready`/`device_ready` from even if hardened) -- both
  authorized explicitly by the auditor as acceptable for this PR.
  Verified: full workspace test suite (1,397 tests across the lib, CLI,
  and `contract_tests` integration binaries) passing, `cargo doc
  --locked --workspace --no-deps` with `RUSTDOCFLAGS="-D warnings"`
  clean (a private intra-doc link this Change introduced broke CI's
  `docs` job on the first push; none of this session's other local
  checks run `cargo doc`, so it was only caught by CI -- fixed and
  reverified locally before repushing), coverage ratchet at 78.91%
  (above the 78.89% baseline), `openspec validate --all --strict`
  77/77, live `magnetar run qwen-test "Hello"` unaffected (the
  production path never calls `warm_model_instance`), and the CI run
  cited above.
- Release artifacts are not final until generated from the exact release commit
  and tag.

### Security Notes

- Cryptographic artifact signing is not implemented for v0.1 and must not be
  claimed as present; it is deferred to a dedicated future change and this
  release makes no authenticated publisher identity claim.
- Cache presence, model alias, local file location, recognized format,
  publisher metadata, source metadata, and fixture markers do not grant trust by
  themselves.
