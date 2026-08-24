# Tasks

## 1. Kernel Scope

- [x] Define Kernel as implementation of Operator.
- [x] Document Kernel versus Operator.
- [x] Document Kernel versus Provider.
- [x] Document Kernel versus Device.
- [x] Document Kernel versus model architecture.
- [x] Document Kernel versus Execution Graph.
- [x] Document Kernel versus Kernel Registry.

## 2. Kernel Module

- [x] Create first-class `kernel` module or equivalent.
- [x] Export canonical kernel types from crate root.
- [x] Keep Kernel Contract platform-neutral.
- [x] Keep Kernel Contract independent from direct client Provider selection.
- [x] Add module-level documentation.

## 3. Kernel Identity

- [x] Define KernelId.
- [x] Include Provider ID.
- [x] Include kernel name.
- [x] Include kernel version.
- [x] Include implemented Operator ID.
- [x] Include implemented Operator version range.
- [x] Include feature flags.
- [x] Include implementation family.
- [x] Include optional build fingerprint.
- [x] Include conformance profile reference.
- [x] Ensure KernelId does not expose function pointers.
- [x] Add identity tests.

## 4. Kernel Advertisement

- [x] Define KernelAdvertisement.
- [x] Include Kernel identity.
- [x] Include implemented Operator.
- [x] Include supported dtypes.
- [x] Include supported layouts.
- [x] Include supported shapes.
- [x] Include supported memory classes.
- [x] Include supported Devices or Device classes.
- [x] Include workspace requirements.
- [x] Include execution modes.
- [x] Include determinism metadata.
- [x] Include precision metadata.
- [x] Include performance hints.
- [x] Include fallback hints.
- [x] Include required Provider features.
- [x] Include required Device features.
- [x] Add advertisement tests.

## 5. Operator Compatibility

- [x] Validate Operator ID compatibility.
- [x] Validate Operator version compatibility.
- [x] Validate operator attributes.
- [x] Validate input arity.
- [x] Validate output arity.
- [x] Validate shape compatibility.
- [x] Validate dtype compatibility.
- [x] Validate layout compatibility.
- [x] Validate memory behavior compatibility.
- [x] Validate determinism compatibility.
- [x] Validate Resource Affinity compatibility.
- [x] Add compatibility tests.

## 6. Shape Constraints

- [x] Define rank requirements.
- [x] Define static dimension requirements.
- [x] Define dynamic dimension support.
- [x] Define alignment requirements.
- [x] Define batch size limits.
- [x] Define sequence length limits.
- [x] Define head count limits.
- [x] Define head dimension limits.
- [x] Define matrix tile constraints.
- [x] Define block size constraints.
- [x] Define page size constraints.
- [x] Define maximum total elements.
- [x] Define maximum total tokens.
- [x] Add shape constraint tests.

## 7. DType Constraints

- [x] Define input dtype support.
- [x] Define output dtype support.
- [x] Define compute dtype support.
- [x] Define accumulation dtype support.
- [x] Define storage dtype support where relevant.
- [x] Define quantized dtype support.
- [x] Define mixed precision support.
- [x] Define conversion requirements.
- [x] Add dtype constraint tests.

## 8. Layout Constraints

- [x] Define contiguous layout support.
- [x] Define strided layout support.
- [x] Define blocked layout support.
- [x] Define packed layout support.
- [x] Define quantized packed layout support.
- [x] Define paged KV cache layout support.
- [x] Define attention-specific layout support.
- [x] Define Provider-owned opaque layout support.
- [x] Define browser-compatible layout support.
- [x] Prevent opaque layout leaking into Component APIs.
- [x] Add layout constraint tests.

## 9. Memory Class Constraints

- [x] Define host memory support.
- [x] Define pinned-host memory support.
- [x] Define device memory support.
- [x] Define unified memory support.
- [x] Define shared memory support.
- [x] Define provider-owned memory support.
- [x] Define browser-linear-memory support.
- [x] Define future WebGPU buffer support.
- [x] Add memory class tests.

## 10. Workspace Requirements

- [x] Define required workspace.
- [x] Define optional workspace.
- [x] Define workspace size formula.
- [x] Define workspace upper bound.
- [x] Define workspace memory class.
- [x] Define workspace alignment.
- [x] Define workspace lifetime.
- [x] Define workspace reuse policy.
- [x] Define per-operation workspace scope.
- [x] Define per-batch workspace scope.
- [x] Define unavailable workspace behavior.
- [x] Add workspace tests.

## 11. Aliasing Behavior

- [x] Define no-aliasing behavior.
- [x] Define input-output alias allowed behavior.
- [x] Define in-place supported behavior.
- [x] Define output aliases input behavior.
- [x] Define internal temporary aliasing.
- [x] Define input mutation behavior.
- [x] Define read-only input behavior.
- [x] Define write-only output behavior.
- [x] Add aliasing tests.

## 12. Resource Affinity

- [x] Declare Provider affinity requirements.
- [x] Declare Device affinity requirements.
- [x] Preserve input Resource Affinity.
- [x] Preserve output Resource Affinity.
- [x] Reject incompatible affinity.
- [x] Require explicit movement or conversion.
- [x] Prevent silent movement.
- [x] Add Resource Affinity tests.

## 13. Execution Modes

- [x] Define synchronous mode.
- [x] Define asynchronous mode.
- [x] Define streamed mode.
- [x] Define batched mode.
- [x] Define graph-captured mode.
- [x] Define provider-fused mode.
- [x] Define browser-compatible mode.
- [x] Define test-fixture mode.
- [x] Add execution mode tests.

## 14. Cancellation

- [x] Define not-supported cancellation.
- [x] Define before-dispatch-only cancellation.
- [x] Define cooperative cancellation.
- [x] Define interruptible cancellation.
- [x] Define timeout-only cancellation.
- [x] Define Provider-specific cancellation metadata.
- [x] Report unsupported cancellation.
- [x] Add cancellation tests.

## 15. Determinism

- [x] Define determinism metadata.
- [x] Include dtype dependency.
- [x] Include Device dependency.
- [x] Include execution mode dependency.
- [x] Include parallel reduction dependency.
- [x] Include accumulation order dependency.
- [x] Include atomic operation dependency.
- [x] Include kernel version dependency.
- [x] Include Provider version dependency.
- [x] Include hardware feature dependency.
- [x] Validate deterministic mode requests.
- [x] Add determinism tests.

## 16. Precision Metadata

- [x] Define accumulation dtype metadata.
- [x] Define rounding mode metadata where known.
- [x] Define approximate math metadata.
- [x] Define fused operation semantics.
- [x] Define tolerance profile.
- [x] Define quantization error profile.
- [x] Define deterministic tolerance profile.
- [x] Add precision metadata tests.

## 17. Fused Kernels

- [x] Define fused kernel metadata.
- [x] Declare operator group implemented.
- [x] Validate fusion preserves graph semantics.
- [x] Reject unsupported fusion.
- [x] Add fused kernel tests.

## 18. Adapter-Aware Kernels

- [x] Declare supported adapter methods.
- [x] Declare maximum adapter rank.
- [x] Declare supported adapter dtypes.
- [x] Declare supported merge/overlay strategy.
- [x] Declare supported target modules.
- [x] Validate active adapter compatibility.
- [x] Add adapter-aware kernel tests.

## 19. KV-Cache-Aware Kernels

- [x] Declare KV cache layout support.
- [x] Declare paged cache support.
- [x] Declare append behavior.
- [x] Declare read behavior.
- [x] Declare cache dtype support.
- [x] Declare cache memory class support.
- [x] Declare Resource Affinity constraints.
- [x] Prevent raw KV cache exposure.
- [x] Add KV-cache-aware kernel tests.

## 20. Prefix-Cache-Aware Behavior

- [x] Support adjusted sequence length.
- [x] Support adjusted context length.
- [x] Support reused prefix boundary metadata.
- [x] Avoid owning Prefix Cache policy.
- [x] Add prefix-related tests.

## 21. Batched Kernels

- [x] Declare max batch size.
- [x] Declare max active sequences.
- [x] Declare max total tokens.
- [x] Declare sequence length constraints.
- [x] Declare padding behavior.
- [x] Declare ragged batch support.
- [x] Declare paged KV cache compatibility.
- [x] Declare per-operation output mapping.
- [x] Declare batch slot compatibility.
- [x] Add batched kernel tests.

## 22. Browser Compatibility

- [x] Keep Kernel Contract platform-neutral.
- [x] Define browser-linear-memory compatible kernels.
- [x] Define JavaScript-mediated execution placeholder.
- [x] Define future WebGPU buffer support.
- [x] Avoid Wasmtime dependency.
- [x] Avoid native Provider loading requirement.
- [x] Return browser-feature-unsupported where needed.
- [x] Add wasm32 check where feasible.

## 23. Kernel Invocation

- [x] Define KernelInvocation.
- [x] Include invocation ID.
- [x] Include Operator invocation reference.
- [x] Include Kernel identity.
- [x] Include input resource references.
- [x] Include output resource references.
- [x] Include workspace reference.
- [x] Include execution mode.
- [x] Include Provider/Device context metadata.
- [x] Include Resource Affinity metadata.
- [x] Include cancellation token.
- [x] Include deadline or timeout.
- [x] Include observability correlation.
- [x] Include policy metadata.
- [x] Prevent Components from creating raw Provider invocations.
- [x] Add invocation tests.

## 24. Kernel Results

- [x] Define KernelResult.
- [x] Include success/failure.
- [x] Include output readiness.
- [x] Include updated resource metadata.
- [x] Include workspace release hints.
- [x] Include timing metadata where available.
- [x] Include determinism metadata.
- [x] Include precision diagnostics.
- [x] Include Provider diagnostics.
- [x] Include Device diagnostics.
- [x] Include structured error.
- [x] Prevent raw handle exposure.
- [x] Add result tests.

## 25. Error Model

- [x] Define kernel-not-found error.
- [x] Define kernel-version-unsupported error.
- [x] Define kernel-Operator-mismatch error.
- [x] Define kernel-attribute-unsupported error.
- [x] Define kernel-shape-unsupported error.
- [x] Define kernel-dtype-unsupported error.
- [x] Define kernel-layout-unsupported error.
- [x] Define kernel-memory-class-unsupported error.
- [x] Define kernel-workspace-unavailable error.
- [x] Define kernel-aliasing-unsupported error.
- [x] Define kernel-Resource-Affinity-conflict error.
- [x] Define kernel-Device-unsupported error.
- [x] Define kernel-Provider-unavailable error.
- [x] Define kernel-Provider-not-ready error.
- [x] Define kernel-Provider-saturated error.
- [x] Define kernel-execution-failed error.
- [x] Define kernel-cancellation-unsupported error.
- [x] Define kernel-cancelled error.
- [x] Define kernel-timeout error.
- [x] Define kernel-determinism-unsupported error.
- [x] Define kernel-precision-unsupported error.
- [x] Define kernel-conformance-failed error.
- [x] Define kernel-browser-feature-unsupported error.
- [x] Define internal-kernel error.

## 26. Conformance

- [x] Define Kernel conformance profile.
- [x] Test Operator semantic correctness.
- [x] Test shape handling.
- [x] Test dtype handling.
- [x] Test layout handling.
- [x] Test memory behavior.
- [x] Test aliasing behavior.
- [x] Test workspace behavior.
- [x] Test Resource Affinity behavior.
- [x] Test cancellation behavior where supported.
- [x] Test determinism claims.
- [x] Test precision tolerance.
- [x] Test error mapping.
- [x] Test observability metadata.
- [x] Add conformance report.

## 27. Fallback

- [x] Define alternate kernel fallback.
- [x] Define alternate Provider fallback.
- [x] Define alternate Device fallback.
- [x] Define explicit dtype conversion fallback.
- [x] Define explicit layout conversion fallback.
- [x] Define host execution fallback.
- [x] Define rejection fallback.
- [x] Prevent silent policy violation.
- [x] Add fallback tests.

## 28. Security And Isolation

- [x] Prevent raw memory exposure.
- [x] Prevent raw model weight exposure.
- [x] Prevent raw prompt exposure.
- [x] Prevent raw KV cache exposure.
- [x] Prevent raw Provider handle exposure.
- [x] Prevent raw Device handle exposure.
- [x] Document trusted native execution risk.
- [x] Add security tests.

## 29. Observability

- [x] Emit kernel advertised observation.
- [x] Emit kernel invocation created observation.
- [x] Emit kernel dispatch started observation.
- [x] Emit kernel dispatch completed observation.
- [x] Emit kernel dispatch failed observation.
- [x] Emit kernel workspace requested observation.
- [x] Emit kernel cancellation requested observation.
- [x] Emit kernel cancelled observation.
- [x] Emit kernel timeout observation.
- [x] Emit kernel fallback considered observation.
- [x] Emit kernel fallback used observation.
- [x] Emit kernel conformance result observation.
- [x] Emit kernel Resource Affinity conflict observation.
- [x] Emit kernel determinism limitation observation.
- [x] Emit kernel precision diagnostic observation.
- [x] Avoid raw tensor/prompt/weight/cache/handle logging.

## 30. Tests

- [x] Test Kernel identity.
- [x] Test Kernel advertisement.
- [x] Test Operator compatibility.
- [x] Test Operator version mismatch.
- [x] Test shape unsupported.
- [x] Test dtype unsupported.
- [x] Test layout unsupported.
- [x] Test memory class unsupported.
- [x] Test workspace unavailable.
- [x] Test aliasing unsupported.
- [x] Test Resource Affinity conflict.
- [x] Test Device unsupported.
- [x] Test Provider not ready.
- [x] Test cancellation unsupported.
- [x] Test deterministic mode unsupported.
- [x] Test fused kernel metadata.
- [x] Test adapter-aware kernel compatibility.
- [x] Test KV-cache-aware kernel compatibility.
- [x] Test batched kernel metadata.
- [x] Test fallback explicitness.
- [x] Test raw handles not exposed.
- [x] Test conformance failure report.

## 31. Documentation

- [x] Document Kernel Contract.
- [x] Document Kernel versus Operator.
- [x] Document Kernel versus Provider.
- [x] Document Kernel identity.
- [x] Document Kernel advertisement.
- [x] Document shape constraints.
- [x] Document dtype constraints.
- [x] Document layout constraints.
- [x] Document memory class constraints.
- [x] Document workspace requirements.
- [x] Document aliasing behavior.
- [x] Document Resource Affinity.
- [x] Document execution modes.
- [x] Document cancellation.
- [x] Document determinism.
- [x] Document precision metadata.
- [x] Document fused kernels.
- [x] Document adapter-aware kernels.
- [x] Document KV-cache-aware kernels.
- [x] Document batched kernels.
- [x] Document conformance.
- [x] Document fallback.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 32. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Kernel tests.
- [x] Run Operator tests.
- [x] Run Execution Graph tests.
- [x] Run Memory Manager tests where impacted.
- [x] Run Provider conformance tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify Kernels implement Operators.
- [x] Verify Kernels are not Providers.
- [x] Verify Kernel metadata is sufficient for future dispatch.
- [x] Verify Memory Manager owns workspace allocation.
- [x] Verify raw handles are not exposed.