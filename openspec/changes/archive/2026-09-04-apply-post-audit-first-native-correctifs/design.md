## Context

The audited code still exposes two distinct notions of "first-native" execution. The intended architecture is Runtime-owned model execution through Model Artifact, Qwen WASM Component, Execution Graph, PreparedExecutionPlan, Kernel Registry, Provider, Runtime-owned tensor/KV resources, Generation, and Sampling. The current implementation still keeps important pieces in `e2e_conformance`, exposes a generation executor that can return logits directly, and lets the CLI reach a fixture helper instead of a production runtime entry point.

Existing specs already define most of the target architecture. This change tightens the contracts around the final vertical slice and supplies an implementation plan to remove remaining bypasses.

## Goals / Non-Goals

**Goals:**

- Make `RuntimeInferenceApi` the only production entry point for CLI generation.
- Ensure logits are produced by executing a ready `ModelInstance` through a compatible `PreparedExecutionPlan`.
- Ensure Qwen operator execution goes through Kernel Registry and Provider dispatch.
- Make the Qwen WASM Component artifact authoritative for E2E graph semantics.
- Implement Runtime-owned prefill plus incremental KV decode for the first baseline.
- Move test fixtures and conformance support out of production behavior.
- Emit evidence from the runtime layers that actually perform work.
- Keep docs and security statements synchronized with implemented behavior.

**Non-Goals:**

- Implement CUDA, Metal, OpenVINO, QNN, WebGPU production, distributed execution, or advanced autotuning.
- Optimize large-model performance beyond correctness required for the first-native baseline.
- Redesign publisher cryptographic identity beyond existing trust policy.
- Introduce Candle or make Qwen a Provider.

## Decisions

1. Introduce a production-facing first-native runtime module before deleting the E2E helper.

   Rationale: the CLI must stop depending on `e2e_conformance` immediately, but the runtime internals can be moved behind a stable facade in smaller steps. The facade starts as a compatibility shell and becomes the owner of the real model executor.

   Alternative considered: rewrite CLI and runtime in one large change. Rejected because it makes review, rollback, and conformance diagnosis harder.

2. Replace `RuntimeGenerationExecutor` with a Runtime-owned model execution engine.

   Rationale: logits are outputs of model execution, not caller inputs. The engine owns `execute_prefill` and `execute_decode_step` over `ModelInstance`, `PreparedExecutionPlan`, and Runtime-managed resources.

   Alternative considered: keep the trait but restrict implementors. Rejected because the trait shape still models arbitrary logits production.

3. Build separate prefill and decode prepared plans.

   Rationale: guards, workload buckets, KV mode, tensor shapes, and Resource Affinity differ between full prompt prefill and one-token decode. Separate plan identities make evidence and invalidation precise.

   Alternative considered: use a single polymorphic plan with loose guards. Rejected because it weakens plan validation and can hide full-history recompute.

4. Keep synthetic logits under test/conformance-only support.

   Rationale: deterministic test fixtures remain useful, but production builds must not expose a seam that can replace model execution.

   Alternative considered: remove all synthetic logits immediately. Rejected because lower-level Sampling and error tests would become needlessly expensive or fragile.

5. Treat E2E evidence as observational.

   Rationale: architecture conformance must be proven by events emitted where work occurs. Boolean evidence assembled after the fact cannot distinguish real execution from a bypass.

   Alternative considered: keep a summarized evidence object. Accepted only if the summary is derived from bounded observations and cannot set facts independently.

## Risks / Trade-offs

- [Risk] The first extraction may preserve legacy internals behind a new facade. -> Mitigation: keep bypass inventory entries removal-required until the implementation is fully moved.
- [Risk] Incremental KV correctness can drift from full-sequence oracle behavior. -> Mitigation: add positional oracle tests comparing prefill+decode logits to full-sequence logits with explicit tolerance.
- [Risk] Disabling direct Reference CPU calls in E2E can reduce debug visibility. -> Mitigation: keep direct Reference CPU calls allowed in unit tests, oracle qualification, and differential tests only.
- [Risk] Wasmtime feature tests may be slower in CI. -> Mitigation: keep default docs/tests fast while making Wasmtime docs/tests explicit quality jobs.
- [Risk] API removal can break existing tests. -> Mitigation: migrate tests to `#[cfg(test)]` or explicit conformance support before removing public exports.

## Migration Plan

1. Add OpenSpec deltas for the post-audit acceptance criteria.
2. Introduce `first_native_runtime` as the public runtime facade and switch CLI callers to it.
3. Move first-native fixture execution from `e2e_conformance` into runtime-owned modules.
4. Add a `RuntimeModelExecutor` implementation that requires ready `ModelInstance` and compatible `PreparedExecutionPlan`.
5. Route Qwen graph nodes through Registry/Provider dispatch and produce per-node observations.
6. Implement prefill KV initialization and decode KV append/update with session/model isolation tests.
7. Gate or remove `RuntimeGenerationExecutor` and `RuntimeGenerationStep::new(logits, ...)` from production APIs.
8. Rebuild E2E around production APIs and observational evidence only.
9. Remove production exports/dependencies for `e2e_conformance`.
10. Run workspace tests, docs, Wasmtime-feature docs/tests, WIT, OpenSpec validation, and quality gates.

Rollback strategy: each step keeps previous conformance fixtures until its replacement passes. Reverting a step must not reintroduce CLI dependence on `e2e_conformance` or production logits injection without restoring the corresponding bypass inventory entry.

## Open Questions

- Whether the final first-native engine should live entirely in `first_native_runtime` or be split into smaller `model_execution`, `qwen_execution`, and `generation_execution` modules.
- Whether synthetic logits test support should be `#[cfg(test)]` only or feature-gated as `synthetic-logits-test-support` for integration tests.
- Which exact artifact format will supply the first checked-in executable Qwen WASM Component fixture.
