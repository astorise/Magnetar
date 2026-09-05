## 1. Immediate Correctness

- [x] 1.1 Fix decode RoPE absolute position in the real generation loop for the token being decoded.
- [x] 1.2 Add instrumentation or test hooks that expose decode token positions without exposing prompt or tensor contents.
- [x] 1.3 Add a multi-step generation-loop oracle test proving prompt length 4 decodes generated tokens at positions 4, 5, and 6.
- [x] 1.4 Verify the RoPE fix with focused generation tests and update any affected static assertions.

## 2. Prepared Plan Execution

- [x] 2.1 Identify all first-native hot-path dispatch sites that still perform per-node kernel selection.
- [x] 2.2 Introduce a prepared plan execution abstraction that consumes graph nodes and published plan bindings.
- [x] 2.3 Route first-native operator dispatch through PlanNodeBinding and PreparedKernelId lookup.
- [x] 2.4 Fail with structured errors when a ready plan is missing, invalidated, stale outside policy, or references a missing prepared kernel.
- [x] 2.5 Add tests proving registry preference changes after plan publication do not alter execution for that plan.
- [x] 2.6 Add tests proving kernel revocation blocks new work while valid in-flight leases can complete.

## 3. Graph-Authoritative Qwen Execution

- [x] 3.1 Add a generic graph executor that walks ExecutionGraph dependencies and binds graph inputs, intermediates, outputs, and resources.
- [x] 3.2 Move Qwen prefill execution from a hard-coded numerical sequence to graph-node execution.
- [x] 3.3 Move Qwen decode execution from a hard-coded numerical sequence to graph-node execution.
- [x] 3.4 Ensure logits consumed by sampling are read from the declared graph output resource.
- [x] 3.5 Add tests for missing bindings, unsupported operators, invalid graphs, removed graph nodes, and graph output provenance.
- [x] 3.6 Remove or confine Qwen helper paths so production first-native cannot compute logits outside the graph executor.

## 4. Qwen Model Component Authority

- [x] 4.1 Define the portable graph representation or builder capability used by the Qwen component.
- [x] 4.2 Update the Qwen WASM component fixture to produce graph semantics for prefill and decode.
- [x] 4.3 Validate component-produced graph semantics before plan preparation.
- [x] 4.4 Fail closed in strict first-native when the component engine, artifact, trust, fuel, deadline, or graph output is invalid.
- [x] 4.5 Replace node-count-only proof with semantic graph comparison and causal evidence.
- [x] 4.6 Run and update WIT/component validation if the component interface changes.

## 5. Runtime Provider And Memory Authority

- [x] 5.1 Remove production first-native local `ReferenceCpuExecutor::new()` execution bypasses.
- [x] 5.2 Resolve the executing provider from Runtime provider registration for each prepared binding.
- [x] 5.3 Add provider submission and completion identities to the Reference CPU execution path.
- [x] 5.4 Remove production first-native local `MemoryManager::default()` allocation bypasses.
- [x] 5.5 Account outputs and workspaces through Runtime MemoryManager.
- [x] 5.6 Add tests for provider removal, mock provider execution, memory limits, output accounting, and workspace release.

## 6. Model Artifact Resource Authority

- [x] 6.1 Represent synthetic first-native fixture weights as model artifact payload resources.
- [x] 6.2 Make model loading create Runtime resources for weights and constants.
- [x] 6.3 Expose ModelInstance weight and constant bindings to graph execution.
- [x] 6.4 Remove production compute reads from `fixture.weights` or equivalent side-channel maps.
- [x] 6.5 Add tests for artifact byte changes, missing weights, digest rejection, per-instance isolation, and unload cleanup.

## 7. Runtime-Owned KV Data

- [x] 7.1 Represent prefill KV state as Runtime-owned resources with session, model instance, layer, affinity, and accounting metadata.
- [x] 7.2 Make decode attention read historical KV through Runtime-owned cache resource bindings.
- [x] 7.3 Move executor-private KV byte storage behind Runtime KV resource APIs.
- [x] 7.4 Implement KV update prepare, commit, abort, and cleanup semantics.
- [x] 7.5 Add tests for sampling failure rollback, provider failure rollback, cancel rollback, double commit, double abort, stale pending state, and wrong-session KV access.

## 8. Causal Evidence And Chat Session

- [x] 8.1 Replace synthesized global evidence booleans with observations emitted by component, graph, plan, provider, resource, KV, sampling, and token commit layers.
- [x] 8.2 Add redaction and bounded-buffer tests for observations.
- [x] 8.3 Route `magnetar chat` turns through the ChatSession Runtime and persistent InferenceSession.
- [x] 8.4 Add chat tests for stable session id across turns, cancellation, close cleanup, KV retention policy, and session isolation.

## 9. Quality, Status, And Final Validation

- [x] 9.1 Move large inline test modules or update coverage configuration so production coverage excludes test-only source.
- [x] 9.2 Re-measure and update the production coverage baseline and documented exclusions.
- [x] 9.3 Reconcile README, OpenSpec notes, and release status so Architecture Freeze #1 remains candidate until P0 evidence passes.
- [x] 9.4 Document cryptographic artifact signatures as deferred to a dedicated future change without authenticated publisher claims.
- [x] 9.5 Run `cargo fmt --all`.
- [x] 9.6 Run `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings`.
- [x] 9.7 Run `cargo test --locked --workspace --all-targets`.
- [x] 9.8 Run WIT/component validation required by changed interfaces.
- [x] 9.9 Run `openspec validate --all --strict` and `git diff --check`.
