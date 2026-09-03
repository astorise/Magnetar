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
- Architecture Freeze #1 is **accepted**: all P0 causal datapath
  requirements (`reach-architecture-freeze-1`) are implemented, test-covered,
  and proven by a full green CI run (formatting, clippy, workspace tests
  across Linux/Windows/macOS, WIT/component validation, wasm32, submodule
  integration, and OpenSpec validation) at commit `0197be1` on
  `make-first-native-datapath-authoritative`, which is now archived as
  `2026-09-03-make-first-native-datapath-authoritative`. `reach-architecture-
  freeze-1`'s own task group 19 re-ran the full causal chain
  (`magnetar run qwen-test "Hello"`, live, not just via test) end to end and
  confirmed evidence for every step from CLI through Model Loading, the real
  Qwen Component, `PreparedExecutionPlan`, the Provider, admitted Tensor
  Resources, Runtime-owned KV Resources, Sampling, and token commit.
- Release artifacts are not final until generated from the exact release commit
  and tag.

### Security Notes

- Cryptographic artifact signing is not implemented for v0.1 and must not be
  claimed as present; it is deferred to a dedicated future change and this
  release makes no authenticated publisher identity claim.
- Cache presence, model alias, local file location, recognized format,
  publisher metadata, source metadata, and fixture markers do not grant trust by
  themselves.
