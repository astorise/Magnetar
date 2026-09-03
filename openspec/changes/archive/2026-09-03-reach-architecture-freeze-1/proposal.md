## Why

Following the first-native datapath authority work (`d9d26f7`, branch
`make-first-native-datapath-authoritative`), a follow-up architecture audit
(2026-09-01) reviewed the resulting implementation against Architecture
Freeze #1. Most findings are implementation gaps against specs that are
already correct: `memory`, `provider`, `kernel-execution-plan`,
`kernel-registry`, `execution-graph`, `model-loading`, `kv-cache`, and
`runtime` already require Provider materialization to follow memory
admission, Provider submit/complete to be causal, `PreparedExecutionPlanExecutor`
to drive production execution, Resource-based generic graph execution, and
transactional KV resources — the code has not caught up yet.

One finding is a genuine spec gap. `model-component`'s `Graph Production`
requirement only says a Model Component SHALL produce graph semantics; it
never defines the portable contract shape a Component uses to do so. Because
that contract does not exist yet, `magnetar-runtime` still synthesizes Qwen's
prefill/decode graphs (`qwen_prefill_graph` / `qwen_decode_graph`) in Rust
instead of sourcing them from the WASM Component, which keeps Qwen-specific
execution semantics inside the Core.

Per this repository's own OpenSpec governance rule (a correct spec plus
non-conformant code is an implementation issue; a new semantic decision or
extension contract is a new Change), only the graph-contract gap justifies
new spec requirements. Everything else is tracked here as an implementation
punch list toward closing Architecture Freeze #1.

## What Changes

- New capability spec `model-component-graph-contract` defining the portable
  contract a Model Component SHALL use to export prefill/decode graph
  semantics to the Runtime: node identities, Operator identities/versions,
  inputs/outputs, Tensor descriptors, weight/constant references, KV logical
  resources, Operator attributes, graph outputs, architecture metadata, and
  contract version. The spec settles the structural choice the audit raised
  (serialized graph descriptor vs. a Runtime-owned graph-builder capability)
  in favor of the Runtime-owned builder, so the Runtime keeps owning
  validation types instead of parsing an opaque blob across the WIT boundary.
- **BREAKING** Once the Qwen Component implements the new contract,
  `magnetar-runtime`'s Rust-synthesized Qwen graph builder is retired from
  the production path. Strict first-native fails closed when no compatible
  Component/Engine is available; there is no Rust-graph fallback.
- A consolidated implementation punch list (`tasks.md`) tracking the
  remaining P0/P1/P2 conformance gaps the audit found against already-correct
  specs: memory admission ordering before Provider materialization, causal
  Provider submit/complete, removal of concrete Provider downcasts from the
  generic dispatch path, `PreparedExecutionPlanExecutor` as the sole
  production authority (no synthetic `KernelCandidate` reconstruction),
  Resource-based (not `HostTensor`-based) generic graph execution,
  Model-Loading-created weight resources (removing the Qwen fixture
  side-channel), transactional KV pending/commit/abort with Provider-side
  release, `ExecutionGraph` producer/consumer topology canonicalization,
  removal of `std::mem::take(MemoryManager)`, per-node causal execution
  evidence, and the submodule/CI follow-through for `components/*`,
  `formats/*`, `providers/*` (submodule wiring already begun on this branch).

## Capabilities

### New Capabilities
- `model-component-graph-contract`: Portable contract a Model Component uses
  to export prefill/decode `ExecutionGraph` semantics to the Runtime,
  replacing ad hoc per-model-family graph builders inside `magnetar-runtime`.

### Modified Capabilities

None. The audit's P0/P1 implementation gaps are conformance issues against
requirements that already exist in `memory`, `provider`,
`kernel-execution-plan`, `kernel-registry`, `execution-graph`,
`model-loading`, `kv-cache`, `runtime`, and `first-native-implementation-cut`
— see `tasks.md` for the tracked punch list and per-item Definition of Done.
If implementation work reveals that one of these specs is actually
insufficient rather than merely unmet, split that item into its own Change
instead of silently drifting the requirement text here.

## Impact

- Affected crates: `magnetar-runtime`, `magnetar-cli`, future
  `components/qwen` (submodule already wired into `.gitmodules` on this
  branch).
- Affected specs: new `openspec/specs/model-component-graph-contract/spec.md`;
  no deltas to existing specs.
- Repository: depends on the `components/qwen`, `components/llama`,
  `formats/gguf`, `formats/safetensors` submodules already added to
  `.gitmodules` on this branch as the eventual home for the Qwen
  graph-contract implementation and other externalized model families.
- Compatibility: production code paths that let `magnetar-runtime` synthesize
  Qwen graph semantics directly will be removed once the Qwen Component
  implements the new contract; test/conformance fixtures must migrate to
  explicit Component-backed fixtures instead of `bind_qwen_fixture_weights()`
  and the Rust graph builder.
