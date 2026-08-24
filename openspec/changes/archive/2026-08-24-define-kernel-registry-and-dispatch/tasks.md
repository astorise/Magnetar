# Tasks

## 1. Registry Scope

- [x] Define Kernel Registry as Runtime-owned metadata index.
- [x] Document Kernel Registry versus Kernel Contract.
- [x] Document Kernel Registry versus Provider.
- [x] Document Kernel Registry versus Scheduler.
- [x] Document Kernel Registry versus Execution Graph.
- [x] Document Kernel Registry versus Kernel Dispatch.

## 2. Dispatch Scope

- [x] Define Kernel Dispatch as Runtime-owned validated execution submission.
- [x] Document Dispatch versus Scheduler planning.
- [x] Document Dispatch versus Provider execution.
- [x] Document Dispatch versus Graph execution.
- [x] Document Dispatch versus raw native function calls.

## 3. Modules

- [x] Create first-class `kernel_registry` module or equivalent.
- [x] Create first-class `kernel_dispatch` module or equivalent.
- [x] Export canonical registry and dispatch types from crate root.
- [x] Keep registry and dispatch platform-neutral.
- [x] Keep registry and dispatch independent from direct client Provider
      selection.
- [x] Add module-level documentation.

## 4. Registry Index

- [x] Index by Operator identity.
- [x] Index by Operator version compatibility.
- [x] Index by Provider identity.
- [x] Index by Device class.
- [x] Index by dtype support.
- [x] Index by layout support.
- [x] Index by shape constraints.
- [x] Index by memory class support.
- [x] Index by execution mode.
- [x] Index by Resource Affinity constraints.
- [x] Index by conformance profile.
- [x] Index by feature flags.
- [x] Add registry index tests.

## 5. Registry Ownership

- [x] Allow Provider kernel advertisements.
- [x] Allow Runtime test fixture kernel advertisements.
- [x] Reject client kernel registration.
- [x] Reject Component kernel registration.
- [x] Validate advertisements before registry insertion.
- [x] Add ownership tests.

## 6. Advertisement Validation

- [x] Validate Provider identity.
- [x] Validate Provider lifecycle.
- [x] Validate Kernel identity.
- [x] Validate implemented Operator identity.
- [x] Validate Operator version compatibility.
- [x] Validate metadata schema.
- [x] Validate dtype metadata.
- [x] Validate layout metadata.
- [x] Validate shape constraints.
- [x] Validate memory class constraints.
- [x] Validate execution mode metadata.
- [x] Validate workspace metadata.
- [x] Validate cancellation metadata.
- [x] Validate determinism metadata.
- [x] Validate precision metadata.
- [x] Validate required Provider features.
- [x] Validate required Device features.
- [x] Validate conformance status where required.
- [x] Add advertisement validation tests.

## 7. Registry Invalidation

- [x] Invalidate entries when Provider unregisters.
- [x] Invalidate entries when Provider fails.
- [x] Invalidate entries when Provider drains.
- [x] Invalidate entries when Provider becomes not ready.
- [x] Invalidate entries when Provider pressure is saturated.
- [x] Invalidate entries when Device disappears.
- [x] Invalidate entries when Device resets.
- [x] Invalidate entries when Device becomes unavailable.
- [x] Invalidate entries when Device memory pressure exceeds policy.
- [x] Invalidate entries when Kernel conformance is revoked.
- [x] Invalidate entries on policy change.
- [x] Invalidate entries on Runtime shutdown.
- [x] Add invalidation tests.

## 8. Kernel Candidate

- [x] Define KernelCandidate.
- [x] Include Kernel identity.
- [x] Include Provider identity.
- [x] Include Device compatibility.
- [x] Include Operator compatibility.
- [x] Include dtype compatibility.
- [x] Include layout compatibility.
- [x] Include shape compatibility.
- [x] Include memory compatibility.
- [x] Include workspace feasibility.
- [x] Include Resource Affinity compatibility.
- [x] Include determinism compatibility.
- [x] Include precision compatibility.
- [x] Include Provider readiness.
- [x] Include Device readiness.
- [x] Include pressure score.
- [x] Include conformance status.
- [x] Include estimated cost.
- [x] Include fallback rank.
- [x] Include rejection reason.
- [x] Add candidate tests.

## 9. Selection Request

- [x] Define KernelSelectionRequest.
- [x] Include request ID.
- [x] Include Operator invocation reference.
- [x] Include graph plan reference.
- [x] Include Model Instance reference where relevant.
- [x] Include input resource references.
- [x] Include output resource requirements.
- [x] Include dtype requirements.
- [x] Include layout requirements.
- [x] Include shape requirements.
- [x] Include memory class requirements.
- [x] Include Resource Affinity requirements.
- [x] Include determinism requirements.
- [x] Include precision requirements.
- [x] Include execution mode preference.
- [x] Include batching metadata.
- [x] Include KV cache metadata.
- [x] Include adapter metadata.
- [x] Include deadline or timeout.
- [x] Include Runtime policy.
- [x] Include observability correlation.
- [x] Prevent Components from bypassing Runtime validation.

## 10. Selection Pipeline

- [x] Implement operator candidate lookup.
- [x] Filter by Operator version.
- [x] Filter by Provider lifecycle.
- [x] Filter by Device lifecycle.
- [x] Filter by shape compatibility.
- [x] Filter by dtype compatibility.
- [x] Filter by layout compatibility.
- [x] Filter by memory class compatibility.
- [x] Check workspace feasibility.
- [x] Validate Resource Affinity.
- [x] Validate determinism and precision.
- [x] Validate batching compatibility.
- [x] Validate adapter compatibility.
- [x] Validate KV cache compatibility.
- [x] Apply conformance gating.
- [x] Apply policy ranking.
- [x] Construct fallback chain.
- [x] Publish selected candidate.
- [x] Add pipeline tests.

## 11. Policy Ranking

- [x] Rank by policy preferences.
- [x] Rank by Provider readiness.
- [x] Rank by Provider pressure.
- [x] Rank by Device readiness.
- [x] Rank by Device pressure.
- [x] Rank by memory pressure.
- [x] Rank by expected latency.
- [x] Rank by expected throughput.
- [x] Rank by workspace cost.
- [x] Rank by data movement cost.
- [x] Rank by layout conversion cost.
- [x] Rank by dtype conversion cost.
- [x] Rank by determinism.
- [x] Rank by precision.
- [x] Rank by conformance profile.
- [x] Rank by failure history.
- [x] Ensure hard Resource Affinity is not overridden.
- [x] Add ranking tests.

## 12. Resource Affinity

- [x] Preserve Provider-bound Resource Affinity.
- [x] Preserve Device-bound Resource Affinity.
- [x] Reject incompatible Kernel candidate.
- [x] Require explicit movement.
- [x] Require explicit conversion.
- [x] Require explicit rebuild where applicable.
- [x] Prevent hidden movement.
- [x] Add Resource Affinity tests.

## 13. Memory Manager Integration

- [x] Check input residency.
- [x] Check output allocation feasibility.
- [x] Check workspace allocation feasibility.
- [x] Check memory class compatibility.
- [x] Check staging requirements.
- [x] Check temporary layout conversion memory.
- [x] Check temporary dtype conversion memory.
- [x] Account for provider-owned memory.
- [x] Account for browser memory limits.
- [x] Apply memory pressure policy.
- [x] Support pending allocation policy.
- [x] Add memory integration tests.

## 14. Dispatch Plan

- [x] Define KernelDispatchPlan.
- [x] Include selected Kernel identity.
- [x] Include owning Provider.
- [x] Include target Device metadata.
- [x] Include Operator invocation.
- [x] Include input resource bindings.
- [x] Include output resource bindings.
- [x] Include workspace reservation.
- [x] Include explicit movement steps.
- [x] Include explicit conversion steps.
- [x] Include execution mode.
- [x] Include cancellation support.
- [x] Include timeout/deadline.
- [x] Include fallback chain.
- [x] Include observability correlation.
- [x] Include cleanup behavior.
- [x] Include expected result metadata.
- [x] Prevent raw Provider handles and function pointers.
- [x] Add dispatch plan tests.

## 15. Dispatch Revalidation

- [x] Re-check Provider readiness.
- [x] Re-check Provider admission.
- [x] Re-check Provider pressure.
- [x] Re-check Device readiness.
- [x] Re-check Device availability.
- [x] Re-check memory reservation validity.
- [x] Re-check Resource Affinity.
- [x] Re-check cancellation state.
- [x] Re-check operation lifecycle.
- [x] Re-check session lifecycle.
- [x] Re-check Model Instance lifecycle.
- [x] Re-check policy.
- [x] Fail closed when stale.
- [x] Add revalidation tests.

## 16. Dispatch Lifecycle

- [x] Define planned state.
- [x] Define ready state.
- [x] Define submitted state.
- [x] Define running state.
- [x] Define completed state.
- [x] Define failed state.
- [x] Define cancel-requested state.
- [x] Define cancelled state.
- [x] Define timed-out state.
- [x] Define fallback-pending state.
- [x] Define fallback-running state.
- [x] Define released state.
- [x] Add lifecycle tests.

## 17. Fallback Chain

- [x] Define alternate Kernel fallback.
- [x] Define same Provider different Device fallback.
- [x] Define alternate Provider fallback.
- [x] Define explicit dtype conversion fallback.
- [x] Define explicit layout conversion fallback.
- [x] Define explicit data movement fallback.
- [x] Define host execution fallback.
- [x] Define rejection fallback.
- [x] Preserve Resource Affinity or explicitly transform it.
- [x] Make fallback observable.
- [x] Add fallback chain tests.

## 18. Scheduler Integration

- [x] Allow Scheduler to request dispatch for graph work.
- [x] Prevent Scheduler from selecting raw native function pointers.
- [x] Allow Scheduler to use Kernel metadata.
- [x] Keep final validation in Runtime dispatch.
- [x] Add scheduler integration tests.

## 19. Execution Graph Integration

- [x] Consume Operator invocations from graph plan.
- [x] Resolve Kernel requirements from graph planning.
- [x] Prevent graphs from embedding raw kernel pointers.
- [x] Add graph integration tests.

## 20. Batching Integration

- [x] Validate batched Kernel compatibility.
- [x] Validate batch size.
- [x] Validate active sequences.
- [x] Validate total tokens.
- [x] Validate ragged batch support.
- [x] Validate per-operation output mapping.
- [x] Preserve per-operation results.
- [x] Add batching integration tests.

## 21. Adapter Integration

- [x] Validate active adapter set.
- [x] Validate adapter execution strategy.
- [x] Revalidate if adapter state changes after planning.
- [x] Add adapter dispatch tests.

## 22. KV Cache Integration

- [x] Validate KV cache lifecycle.
- [x] Validate KV cache layout.
- [x] Validate KV cache dtype.
- [x] Validate KV cache memory class.
- [x] Validate KV cache Resource Affinity.
- [x] Replan or fail if KV cache becomes invalid.
- [x] Add KV cache dispatch tests.

## 23. Prefix Cache Integration

- [x] Use adjusted prefill boundaries.
- [x] Use adjusted sequence lengths.
- [x] Use adjusted context lengths.
- [x] Preserve Prefix Cache privacy policy.
- [x] Add Prefix Cache dispatch tests.

## 24. Provider Integration

- [x] Accept Provider Kernel advertisements.
- [x] Reject invalid Provider advertisements.
- [x] Dispatch only Runtime-created Kernel Invocations.
- [x] Re-check Provider status before dispatch.
- [x] Map Provider dispatch errors.
- [x] Add Provider integration tests.

## 25. Device Integration

- [x] Validate Device class.
- [x] Validate Device memory classes.
- [x] Validate Device dtype support.
- [x] Validate Device layout support.
- [x] Validate Device execution limits.
- [x] Validate Device feature flags.
- [x] Re-check Device state before dispatch.
- [x] Add Device integration tests.

## 26. Conformance Gating

- [x] Define conformance-required policy.
- [x] Gate by Operator family.
- [x] Gate by Provider type.
- [x] Gate by production mode.
- [x] Gate by safety level.
- [x] Gate by deterministic mode.
- [x] Gate by precision policy.
- [x] Gate by dynamic Provider trust.
- [x] Gate by test profile.
- [x] Reject Kernel when required conformance is missing or failed.
- [x] Add conformance gating tests.

## 27. Dispatch Result

- [x] Define KernelDispatchResult.
- [x] Include selected Kernel identity.
- [x] Include Provider identity.
- [x] Include Device metadata.
- [x] Include success/failure.
- [x] Include output readiness.
- [x] Include updated resource metadata.
- [x] Include timing metadata.
- [x] Include fallback used.
- [x] Include cancellation result.
- [x] Include determinism metadata.
- [x] Include precision diagnostics.
- [x] Include Provider diagnostics.
- [x] Include Device diagnostics.
- [x] Include structured error.
- [x] Prevent raw handle/value exposure.
- [x] Add result tests.

## 28. Error Model

- [x] Define kernel-registry-unavailable error.
- [x] Define kernel-advertisement-invalid error.
- [x] Define kernel-registration-denied error.
- [x] Define kernel-candidate-not-found error.
- [x] Define kernel-candidate-incompatible error.
- [x] Define kernel-selection-failed error.
- [x] Define kernel-policy-denied error.
- [x] Define kernel-conformance-required error.
- [x] Define kernel-conformance-missing error.
- [x] Define kernel-conformance-failed error.
- [x] Define kernel-Provider-unavailable error.
- [x] Define kernel-Provider-not-ready error.
- [x] Define kernel-Provider-saturated error.
- [x] Define kernel-Device-unavailable error.
- [x] Define kernel-Device-incompatible error.
- [x] Define kernel-Device-lost error.
- [x] Define kernel-memory-infeasible error.
- [x] Define kernel-workspace-unavailable error.
- [x] Define kernel-Resource-Affinity-conflict error.
- [x] Define kernel-dispatch-plan-invalid error.
- [x] Define kernel-dispatch-stale error.
- [x] Define kernel-dispatch-rejected error.
- [x] Define kernel-dispatch-failed error.
- [x] Define kernel-fallback-unavailable error.
- [x] Define kernel-fallback-failed error.
- [x] Define kernel-cancellation-unsupported error.
- [x] Define kernel-cancelled error.
- [x] Define kernel-timeout error.
- [x] Define kernel-browser-feature-unsupported error.
- [x] Define internal-kernel-registry error.
- [x] Define internal-kernel-dispatch error.

## 29. Observability

- [x] Emit kernel advertisement received observation.
- [x] Emit kernel advertisement accepted observation.
- [x] Emit kernel advertisement rejected observation.
- [x] Emit kernel registry updated observation.
- [x] Emit kernel registry invalidated observation.
- [x] Emit kernel candidate lookup observation.
- [x] Emit kernel candidate rejected observation.
- [x] Emit kernel candidate ranked observation.
- [x] Emit kernel selected observation.
- [x] Emit dispatch plan created observation.
- [x] Emit dispatch plan revalidated observation.
- [x] Emit dispatch submitted observation.
- [x] Emit dispatch running observation.
- [x] Emit dispatch completed observation.
- [x] Emit dispatch failed observation.
- [x] Emit fallback considered observation.
- [x] Emit fallback selected observation.
- [x] Emit fallback failed observation.
- [x] Emit conformance gating applied observation.
- [x] Emit memory feasibility failed observation.
- [x] Emit Resource Affinity conflict observation.
- [x] Emit Provider pressure affected selection observation.
- [x] Emit Device pressure affected selection observation.
- [x] Avoid raw tensor/prompt/weight/cache/handle/pointer logging.

## 30. Browser Compatibility

- [x] Keep registry/dispatch platform-neutral.
- [x] Support browser-compatible Provider metadata.
- [x] Support WebAssembly linear memory metadata.
- [x] Support JavaScript-mediated execution placeholder.
- [x] Support future WebGPU buffer metadata.
- [x] Avoid Wasmtime dependency.
- [x] Avoid native Provider loading requirement.
- [x] Return unsupported browser errors where needed.
- [x] Add wasm32 check where feasible.

## 31. Tests

- [x] Test Provider Kernel advertisement accepted.
- [x] Test invalid advertisement rejected.
- [x] Test Component registration rejected.
- [x] Test candidate lookup by Operator.
- [x] Test Operator version filtering.
- [x] Test Provider not ready filtering.
- [x] Test Device unavailable filtering.
- [x] Test dtype filtering.
- [x] Test layout filtering.
- [x] Test shape filtering.
- [x] Test Resource Affinity conflict.
- [x] Test Memory Manager workspace infeasible.
- [x] Test conformance gating missing.
- [x] Test policy ranking.
- [x] Test dispatch plan creation.
- [x] Test dispatch stale revalidation.
- [x] Test fallback alternate kernel.
- [x] Test fallback explicit conversion.
- [x] Test fallback rejection.
- [x] Test Provider dispatch failure mapping.
- [x] Test batched dispatch compatibility.
- [x] Test adapter state revalidation.
- [x] Test KV cache invalid before dispatch.
- [x] Test raw function pointer not exposed.
- [x] Test raw Provider handle not exposed.

## 32. Documentation

- [x] Document Kernel Registry.
- [x] Document Kernel Dispatch.
- [x] Document registry ownership.
- [x] Document advertisement validation.
- [x] Document registry invalidation.
- [x] Document Kernel Candidate.
- [x] Document selection request.
- [x] Document selection pipeline.
- [x] Document policy ranking.
- [x] Document Resource Affinity preservation.
- [x] Document Memory Manager integration.
- [x] Document Dispatch Plan.
- [x] Document Dispatch lifecycle.
- [x] Document fallback chain.
- [x] Document Scheduler relationship.
- [x] Document Execution Graph relationship.
- [x] Document Batching relationship.
- [x] Document Adapter relationship.
- [x] Document KV Cache relationship.
- [x] Document Provider/Device relationship.
- [x] Document conformance gating.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 33. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Kernel Registry tests.
- [x] Run Kernel Dispatch tests.
- [x] Run Kernel tests.
- [x] Run Operator tests.
- [x] Run Execution Graph tests.
- [x] Run Scheduler tests.
- [x] Run Memory Manager tests.
- [x] Run Provider conformance tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify registry is Runtime-owned.
- [x] Verify dispatch is Runtime-created.
- [x] Verify Provider/Device selection is not client-authored.
- [x] Verify Resource Affinity is never silently overridden.
- [x] Verify no raw function pointers or handles are exposed.
