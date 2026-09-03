## Context

The previous first-native corrective pass made the incremental decode path safer and improved lifecycle handling, but the newest audit identifies a deeper architectural gap: the Runtime can validate Component, graph, plan, provider, memory, model, and KV abstractions while the production first-native path still contains model-specific shortcuts. Architecture Freeze #1 must remain a candidate until those abstractions causally drive execution.

The target path is a local CPU first-native baseline. Magnetar remains the local execution authority; Components define portable model semantics; Runtime validates and prepares those semantics; Providers own native execution details; Devices and raw handles remain hidden from Components and Runtime core.

## Goals / Non-Goals

**Goals:**

- Execute first-native Qwen through `ExecutionGraph` plus published `PreparedExecutionPlan` bindings.
- Remove normal hot-path kernel rediscovery and per-node reselection after a plan generation is ready.
- Resolve Provider execution and memory allocation/accounting through Runtime-owned registries and managers.
- Bind weights and KV state to Runtime resources rather than fixture maps or executor-private stores.
- Correct decode RoPE absolute positions in the real generation loop.
- Produce causal, bounded, redacted evidence from the layers that performed each action.
- Keep `magnetar chat` attached to one persistent Runtime inference session.
- Reconcile coverage and Architecture Freeze status once the P0 datapath evidence exists.

**Non-Goals:**

- No new hardware provider, provider selection policy, model hub, production server, dynamic native provider loading, or AI kernel generation.
- No authenticated artifact publisher identity until a separate cryptographic signature change defines it.
- No Tachyon dependency for standalone local inference.

## Decisions

1. Introduce a generic prepared graph execution path before deleting Qwen helpers.

   The first implementation will add a `PreparedExecutionPlanExecutor` shape that consumes an `ExecutionGraph`, a plan generation, graph inputs, ModelInstance resource bindings, Runtime memory, Runtime provider resolution, and optional KV bindings. This lets Qwen migrate node by node while keeping tests narrow. The alternative, rewriting Qwen compute and graph/component semantics in one step, has higher regression risk and makes failures harder to isolate.

2. Treat the Qwen component as the semantic graph producer for strict first-native.

   The component can either return a portable serialized graph or drive a Runtime-owned graph-builder capability. The Runtime remains responsible for validation, resource binding, provider/device selection, and plan preparation. The alternative, continuing to build Rust graphs and compare component node counts, proves participation but not authority.

3. Keep plan selection and execution as separate phases.

   Kernel registry selection is allowed while preparing or regenerating a plan. Execution of a ready plan must look up the published `PlanNodeBinding` and `PreparedKernelId`, validate guards, resolve the bound Provider/Device, and submit through that binding. Fallbacks require explicit plan invalidation or a new plan generation.

4. Make resources the crossing point between subsystems.

   Model weights, intermediates, logits, workspaces, and KV cache entries will be represented as Runtime resource identities with affinity and accounting. Providers can own storage bytes internally, but Runtime owns identity, lifecycle, compatibility checks, and memory accounting.

5. Make KV updates transactional at the Runtime boundary.

   Decode may prepare new KV state before sampling, but committed cache state changes only after sampling and token commit succeed. Failure, cancellation, timeout, or session close aborts pending updates. This preserves the generation invariant without requiring Provider-native cancellation guarantees.

6. Fix RoPE decode position independently first.

   The real generation loop currently has an isolated off-by-one risk when decoding `generated_tokens.last()`. This is corrected before the broader executor migration so later oracle failures are not polluted by a known position bug.

## Risks / Trade-offs

- Broad datapath rewrite -> Phase the implementation behind tests that prove one invariant at a time, starting with RoPE and plan execution.
- Component graph API/WIT churn -> Prefer an internal portable graph representation first, then extend WIT only where the existing component boundary requires it.
- Resource accounting may expose missing lifecycle states -> Add structured errors and cleanup tests before enabling stricter enforcement in more call sites.
- Provider abstraction may not yet expose all execution evidence needed -> Add minimal submission/completion identities to the Reference CPU path before generalizing.
- Moving tests for coverage can cause noisy diffs -> Defer coverage relocation until production behavior is closed and keep pure moves separate from semantic edits.

## Migration Plan

1. Correct decode RoPE position and add multi-step generation-loop oracle coverage.
2. Add prepared plan execution primitives and tests proving no hot-path reselection.
3. Route Qwen execution through graph nodes and plan bindings while retaining temporary compatibility adapters only inside tests.
4. Move provider resolution and memory accounting to Runtime-owned objects.
5. Move weights and KV state behind ModelInstance/KV resource bindings.
6. Replace synthesized evidence with causal observations from graph/plan/provider/resource/KV/sampling layers.
7. Update CLI chat session behavior, coverage scope, and Freeze status documentation.
8. Run full validation: formatting, clippy, tests, OpenSpec, WIT/component checks, and coverage.

Rollback is branch-level: each phase is test-backed and can be reverted before merge if it regresses baseline CPU-local inference.
