## Why

The first-native CPU baseline still validates the intended Runtime abstractions while parts of the actual inference datapath can be driven by parallel Rust helpers, local executors, local memory managers, fixture weights, and hidden KV state. This change closes that gap before Architecture Freeze #1 is accepted by making the Model Component, ExecutionGraph, PreparedExecutionPlan, Runtime ProviderLoader, Runtime MemoryManager, ModelInstance resources, Runtime-owned KV cache, and sampling path causally responsible for generation.

## What Changes

- Make the first-native hot path execute the published `PreparedExecutionPlan` bindings without per-node kernel rediscovery or normal reselection.
- Make `ExecutionGraph` the authoritative numerical recipe for Qwen prefill/decode execution, with graph outputs feeding logits and sampling.
- Make the Qwen WASM Model Component produce or describe the portable graph semantics used by Runtime validation and planning, instead of proving participation through node-count evidence.
- Route first-native compute through the Provider registered in Runtime and the Runtime-owned `MemoryManager`; remove local production `ReferenceCpuExecutor::new()` and `MemoryManager::default()` bypasses from the compute path.
- Bind executed weights to resources created by model artifact loading and referenced by the active `ModelInstance`; remove fixture-weight side channels from production compute.
- Store and update real KV cache data as Runtime-owned resources with accounting, affinity, lifecycle, and transactional commit/abort semantics.
- Correct decode RoPE absolute positions in the real generation loop and cover multi-step decode with a full generation-loop oracle.
- Replace synthesized global evidence booleans with causal observations emitted by graph, plan, provider submission/completion, tensor resource, KV, sampling, and token commit layers.
- Ensure `magnetar chat` uses one persistent Runtime inference session across turns, with cancellation and close acting on that session.
- Reconcile Architecture Freeze status and production coverage evidence after the P0 datapath is closed.
- Defer cryptographic artifact signatures to a future dedicated change; do not claim authenticated publisher identity in this cycle.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `first-native-execution-profile`: require the baseline profile to prove the complete causal local inference path, not just contract participation.
- `execution-graph`: make the graph the authoritative execution recipe for model compute.
- `kernel-execution-plan`: require execution through immutable published plan bindings and prepared kernel identities.
- `model-component`: require Qwen component graph semantics production or description under Runtime validation.
- `provider`: require first-native compute to resolve and execute the registered Runtime provider.
- `memory`: require compute outputs, workspaces, weights, and KV resources to be allocated/accounted by the Runtime memory manager.
- `model-loading`: require executed model weights to originate from loaded model artifact resources.
- `model-instance`: require active instances to expose stable resource bindings for weights and constants used by execution.
- `kv-cache`: require real KV bytes/resources to be Runtime-owned and transactionally updated.
- `generation`: require decode steps to carry correct absolute positions and commit tokens only after successful causal execution.
- `inference-api`: require generation, chat, cancellation, and session lifecycle to use persistent Runtime sessions.
- `observability`: require causal per-node/per-submission evidence and bounded redacted observations.
- `quality`: require production-only coverage scope to exclude test-only code from measured production source.
- `project-architecture`: require Architecture Freeze #1 status to remain candidate until the P0 causal datapath is proven.

## Impact

- Affected crates: primarily `magnetar-runtime`, with possible WIT/component test fixture updates for Qwen graph production.
- Affected OpenSpec contracts: execution graph, execution plan, model component/loading/instance, provider, memory, KV cache, generation, inference API, observability, quality, and architecture status.
- Affected CLI behavior: `magnetar chat` must execute turns on its persistent Runtime session.
- Compatibility: no new hardware backend or provider selection policy is introduced; existing contracts are tightened so first-native execution must use the Runtime-owned path.
- Non-goals: CUDA, Metal, OpenVINO, QNN, WebGPU, production HTTP serving, model hub downloads, AI kernel generation, hot-loaded provider binaries, and artifact signature implementation.
