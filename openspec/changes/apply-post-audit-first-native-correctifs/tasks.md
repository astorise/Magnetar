## 1. OpenSpec Contract Updates

- [x] 1.1 Create the post-audit OpenSpec change with proposal, design, and delta specs.
- [x] 1.2 Add first-native generation, plan, registry, Qwen component, inference API, CLI, E2E, implementation-cut, quality, and release-security deltas.
- [x] 1.3 Validate the OpenSpec change with strict validation.

## 2. Runtime-Owned First-Native Execution

- [x] 2.1 Add a production-facing `first_native_runtime` facade and switch CLI callers away from direct `e2e_conformance` helpers.
- [x] 2.2 Move first-native success-path implementation out of `e2e_conformance` into runtime-owned modules.
- [x] 2.3 Introduce a Runtime model execution engine with `execute_prefill` and `execute_decode_step` responsibilities.
- [x] 2.4 Require a ready `ModelInstance` for first-native generation and fail without one.
- [x] 2.5 Require a compatible `PreparedExecutionPlan` for each prefill/decode step and fail without one.
- [x] 2.6 Reject invalidated plans for new first-native work.

## 3. Kernel Registry And Provider Dispatch

- [x] 3.1 Register every first-native Qwen required operator kernel in the Kernel Registry.
- [x] 3.2 Bind each Qwen graph node to KernelId, implementation identity, Provider, Device, and PreparedKernelId.
- [x] 3.3 Execute Qwen E2E model operators only through Registry and Provider dispatch.
- [x] 3.4 Add a regression test that disabling a required kernel fails planning/execution instead of bypassing.
- [x] 3.5 Restrict direct Reference CPU function calls to unit, qualification, oracle, or differential tests.

## 4. Qwen WASM Component Authority

- [x] 4.1 Add or generate an executable Qwen WASM Component Artifact fixture.
- [x] 4.2 Validate Component Artifact trust before first-native planning.
- [x] 4.3 Instantiate the Qwen Component through the configured Component Runtime with Wasmtime limits active.
- [ ] 4.4 Build the executed graph from the Component output and validate it before planning.
- [x] 4.5 Add failure tests for missing artifact, digest mismatch, trust rejection, fuel exhaustion, deadline, invalid output, incompatible graph, and no Provider authority.

## 5. Incremental KV Decode

- [ ] 5.1 Implement prefill KV creation/population as Runtime-owned Session and ModelInstance state.
- [ ] 5.2 Implement decode that consumes existing KV and submits only newly admitted token input for the baseline.
- [ ] 5.3 Append/update KV for new K/V values with correct completion dependencies.
- [ ] 5.4 Ensure RoPE and attention use absolute position and historical plus new KV.
- [ ] 5.5 Add non-recompute, KV dependency invalidation, session isolation, and full-sequence oracle comparison tests.
- [ ] 5.6 Release KV state cleanly on Session close/cancel/unload according to policy.

## 6. Remove Production Logits Injection

- [x] 6.1 Inventory all `RuntimeGenerationExecutor` and `RuntimeGenerationStep::new(logits, ...)` uses.
- [ ] 6.2 Replace production generation execution with the Runtime model execution engine.
- [x] 6.3 Move synthetic logits support under `#[cfg(test)]` or an explicit non-production conformance feature.
- [x] 6.4 Remove or gate public API exports that permit normal callers to inject logits.
- [x] 6.5 Add a static regression check forbidding the legacy seam outside allowed modules.

## 7. CLI Runtime Cutover

- [ ] 7.1 Resolve `model_ref` through production model resolution for `run`, `chat`, `serve`, and agent generation.
- [ ] 7.2 Load or reuse the resolved Model Artifact and create/reuse a ready ModelInstance.
- [ ] 7.3 Create Runtime Inference Sessions and submit prompt/generation requests through RuntimeInferenceApi only.
- [x] 7.4 Map model-not-found, artifact-invalid, trust-rejected, load-failed, component-load-failed, provider-unavailable, plan-unavailable, generation-failed, and generation-cancelled errors.
- [x] 7.5 Add CLI tests proving distinct model_ref values are not ignored and no CLI code calls Provider/Kernel/Reference CPU APIs.

## 8. Runtime-Owned Evidence And E2E Isolation

- [ ] 8.1 Emit bounded observations from Component validation, Component instantiation, ModelInstance readiness, graph validation, plan selection, guard acceptance, Kernel resolution, Kernel preparation, Provider submission/completion, KV commits, logits production, Sampling, and token commit.
- [ ] 8.2 Make E2E collect and verify observations instead of fabricating boolean evidence.
- [ ] 8.3 Correlate observations with request, Session, ModelInstance, PlanGeneration, GraphNode, Kernel, Provider, Device, Submission, Completion, and KV position identities.
- [x] 8.4 Make `e2e_conformance` test-only or non-production and remove production exports/dependencies.
- [ ] 8.5 Add architectural regression tests for no direct Reference CPU bypass, no logits injection, no full-history decode, mandatory Component, and CLI independence.

## 9. Documentation, Security, And Quality

- [x] 9.1 Update `SECURITY.md` so implemented Wasmtime fuel, deadline, resource-limit, no-ambient-WASI, and trust controls are not listed as absent.
- [x] 9.2 Verify `cargo doc --workspace --no-deps`.
- [x] 9.3 Verify `cargo doc -p magnetar-runtime --no-deps --features wasmtime-component-engine`.
- [x] 9.4 Ensure CI explicitly runs the docs quality gates and blocks merge on failure.
- [x] 9.5 Update architecture docs after the runtime extraction is complete.

## 10. Final Validation

- [x] 10.1 Run formatting, workspace tests, contract tests, E2E first-native, Wasmtime feature tests, WIT validation, OpenSpec validation, cargo-deny, and coverage gate.
- [x] 10.2 Clear or downgrade bypass inventory entries only after code removal is verified.
- [ ] 10.3 Confirm Architecture Freeze #1 criteria: F01-F06 complete, authoritative E2E green, direct bypass count zero, incremental KV proven, CLI uses RuntimeInferenceApi, and Qwen WASM participates in the System Under Test.

## Implementation Pause Note

The change is validated and partially implemented. The 16 remaining unchecked tasks are
intentionally left open because they require the production Runtime cutover that
is still absent from the codebase: Qwen WASM Component output exposed as an
execution graph rather than only scalar graph counters, incremental KV decode
owned by Runtime Session/ModelInstance state, replacement of the remaining
crate-internal RuntimeGenerationExecutor execution path, CLI model_ref
resolution/load/session cutover through production RuntimeInferenceApi, and
runtime-owned observation/correlation evidence.
