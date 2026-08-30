# Tasks

## 1. Freeze The First Profile

- [x] Accept Change 62 as mandatory first-profile scope.
- [x] Accept Change 63 as implementation cut.
- [x] Record Architecture Freeze #1.
- [x] Document rules for reopening architecture.
- [x] Prevent deferred capabilities from blocking implementation.

## 2. Migration Inventory

- [x] Locate caller-provided `next_logits`.
- [x] Locate equivalent forward callbacks.
- [x] Locate CLI fake/placeholder logits.
- [x] Locate direct Reference CPU E2E calls.
- [x] Locate full-sequence decode shortcuts.
- [x] Locate Candle paths that could satisfy tests accidentally.
- [x] Classify each path as remove/deprecate/isolate.
- [x] Track removal before final cut.

## 3. Runtime Foundations

- [x] Stabilize ProviderId.
- [x] Stabilize DeviceId.
- [x] Stabilize TensorResourceId.
- [x] Stabilize PreparedKernelId.
- [x] Stabilize ExecutionPlanId/generation.
- [x] Stabilize ExecutionStream baseline.
- [x] Stabilize CompletionToken baseline.
- [x] Stabilize structured execution errors.

## 4. Tensor Foundations

- [x] Implement TensorDescriptor.
- [x] Implement shape validation.
- [x] Implement overflow-safe element count.
- [x] Implement byte-size validation.
- [x] Implement f32 baseline.
- [x] Implement dense contiguous layout.
- [x] Implement TensorResource.
- [x] Implement minimal ResourceView.
- [x] Add positive/negative tests.

## 5. Memory Foundations

- [x] Route Tensor storage through Memory Manager.
- [x] Implement simple pool/arena.
- [x] Support alignment.
- [x] Support persistent allocations.
- [x] Support transient allocations.
- [x] Support simple reuse.
- [x] Protect in-flight lifetime.
- [x] Add allocation-failure tests.

## 6. Operator Catalog

- [x] Implement embedding Operator validation.
- [x] Implement matmul Operator validation.
- [x] Implement rmsnorm Operator validation.
- [x] Implement rope Operator validation.
- [x] Implement attention Operator validation.
- [x] Implement softmax Operator validation.
- [x] Implement silu Operator validation.
- [x] Implement add Operator validation.
- [x] Implement mul Operator validation.
- [x] Implement residual-add Operator validation.
- [x] Implement dtype-conversion Operator validation.
- [x] Implement layout-conversion Operator validation.

## 7. Basic Reference CPU Kernels

- [x] Implement add.
- [x] Implement mul.
- [x] Implement residual-add.
- [x] Implement silu.
- [x] Implement rmsnorm.
- [x] Add independent mathematical references.
- [x] Add numerical tolerance tests.

## 8. MatMul And Embedding Kernels

- [x] Implement f32 MatMul.
- [x] Implement rectangular matrices.
- [x] Validate incompatible shapes.
- [x] Implement Embedding.
- [x] Validate token bounds.
- [x] Add deterministic fixtures.

## 9. RoPE Kernel

- [x] Implement position zero.
- [x] Implement non-zero position.
- [x] Implement multi-position prefill.
- [x] Implement incremental decode position.
- [x] Add regression tests.

## 10. Softmax Kernel

- [x] Implement numerically stable softmax.
- [x] Handle large logits.
- [x] Define NaN behavior.
- [x] Add golden tests.

## 11. Attention Kernel

- [x] Implement causal Attention.
- [x] Implement Qwen fixture head geometry.
- [x] Implement grouped-query KV mapping.
- [x] Consume prior KV state.
- [x] Add prefill tests.
- [x] Add incremental decode tests.

## 12. Conversion Kernels

- [x] Implement required dtype-conversion baseline.
- [x] Implement required layout-conversion baseline.
- [x] Avoid fake no-op conversion when semantics differ.
- [x] Add tests.

## 13. Reference CPU Provider

- [x] Expose logical CPU Device.
- [x] Register capabilities.
- [x] Prepare Kernels.
- [x] Execute Prepared Kernels.
- [x] Return structured errors.
- [x] Implement synchronous ExecutionStream.
- [x] Return completed CompletionToken.
- [x] Add Provider conformance.

## 14. Kernel Registry

- [x] Register all mandatory CPU Kernels.
- [x] Reject invalid registration.
- [x] Surface duplicate/conflict errors.
- [x] Query candidates.
- [x] Apply eligibility.
- [x] Select deterministic candidate.
- [x] Add Registry tests.

## 15. Dispatch

- [x] Dispatch selected Kernel to Provider.
- [x] Use PreparedKernelId.
- [x] Preserve Tensor validation.
- [x] Preserve Memory Manager authority.
- [x] Add dispatch tests.

## 16. Synthetic Registry Integration

- [x] Build small synthetic Operator workload.
- [x] Execute via Registry.
- [x] Execute via Provider.
- [x] Verify numerical result.
- [x] Verify no direct Kernel bypass.

## 17. Execution Graph Executor

- [x] Execute multi-node Operator graph.
- [x] Resolve dependencies.
- [x] Bind Tensor Resources.
- [x] Propagate structured failures.
- [x] Add graph tests.

## 18. Allocation Plan Baseline

- [x] Build conservative AllocationPlan.
- [x] Bind persistent slots.
- [x] Bind transient slots.
- [x] Bind workspace slots.
- [x] Support safe reuse.
- [x] Add Plan-memory tests.

## 19. Prepared Execution Plan

- [x] Build Plan from graph.
- [x] Bind Kernel IDs.
- [x] Bind PreparedKernel IDs.
- [x] Bind CPU Device.
- [x] Bind resource slots.
- [x] Bind execution order.
- [x] Bind AllocationPlan.
- [x] Add Plan guards.
- [x] Add Plan lifecycle tests.

## 20. Prepared Plan Reuse

- [x] Execute same Plan repeatedly.
- [x] Reuse Kernel bindings.
- [x] Reuse compatible resource slots.
- [x] Verify Registry full resolution not repeated unnecessarily.
- [x] Add instrumentation.

## 21. Execution Stream Baseline

- [x] Implement one logical CPU compute stream.
- [x] Support submission.
- [x] Support dependencies.
- [x] Produce CompletionToken.
- [x] Update ResourceReadiness.
- [x] Add synchronous stream tests.

## 22. Deterministic Qwen Fixture

- [x] Freeze fixture version.
- [x] Freeze model configuration.
- [x] Freeze deterministic weights.
- [x] Freeze tokenizer.
- [x] Record artifact digests.
- [x] Keep fixture CI-sized.

## 23. Model Artifact

- [x] Parse fixture config.
- [x] Parse/load weight file.
- [x] Validate tensor names.
- [x] Validate shapes.
- [x] Validate dtype.
- [x] Normalize into Model Artifact.
- [x] Add malformed artifact tests.

## 24. Model Artifact Golden Data

- [x] Freeze config digest.
- [x] Freeze weights digest.
- [x] Freeze tokenizer digest.
- [x] Freeze Model Artifact identity.
- [x] Add integrity tests.

## 25. Qwen WASM Component Artifact

- [x] Produce Component Artifact.
- [x] Instantiate via Platform Component Engine.
- [x] Use Wasmtime implementation.
- [x] Keep Wasmtime types private.
- [x] Add Component load tests.

## 26. Qwen Config Interpretation

- [x] Read normalized Qwen fixture config.
- [x] Validate hidden size.
- [x] Validate head count.
- [x] Validate KV head count.
- [x] Validate intermediate size.
- [x] Validate vocabulary.
- [x] Add invalid config tests.

## 27. Qwen Graph Construction

- [x] Build embedding path.
- [x] Build layer RMSNorm.
- [x] Build Q/K/V projections.
- [x] Build RoPE.
- [x] Build Attention.
- [x] Build output projection.
- [x] Build residual.
- [x] Build gated MLP.
- [x] Build final RMSNorm.
- [x] Build LM head.
- [x] Add graph structure tests.

## 28. Component Authority Boundary

- [x] Prevent concrete Provider selection.
- [x] Prevent concrete Device selection.
- [x] Prevent native memory access.
- [x] Prevent native Kernel handle access.
- [x] Prevent ambient network.
- [x] Prevent ambient filesystem.
- [x] Add authority tests.

## 29. Model Loading

- [x] Load Model Artifact.
- [x] Instantiate Qwen Component.
- [x] Resolve logical weights.
- [x] Resolve Reference CPU Provider.
- [x] Build Execution Graph.
- [x] Build Prepared Plan.
- [x] Produce Model Instance.
- [x] Add lifecycle tests.

## 30. Model Instance Readiness

- [x] Require valid Artifact.
- [x] Require Component ready.
- [x] Require Provider ready.
- [x] Require required Kernels.
- [x] Require Prepared Plan.
- [x] Add readiness failures.

## 31. KV Cache Storage

- [x] Implement per-layer K storage.
- [x] Implement per-layer V storage.
- [x] Implement sequence length.
- [x] Implement Session ownership.
- [x] Implement bounds.
- [x] Add storage tests.

## 32. KV Append

- [x] Append K for correct layer/position.
- [x] Append V for correct layer/position.
- [x] Increment logical length correctly.
- [x] Add append tests.

## 33. KV Read

- [x] Read prior K.
- [x] Read prior V.
- [x] Respect current sequence length.
- [x] Add historical-state tests.

## 34. Session Isolation

- [x] Prevent KV sharing between unrelated Sessions.
- [x] Support independent positions.
- [x] Add concurrent Session tests where practical.

## 35. Prefill

- [x] Tokenize prompt.
- [x] Execute real Qwen graph.
- [x] Populate KV.
- [x] Produce prefill logits.
- [x] Freeze golden values.
- [x] Add prefill E2E at Runtime model layer.

## 36. Incremental Decode

- [x] Accept newly generated/input token.
- [x] Reuse prior KV.
- [x] Use current sequence position.
- [x] Execute Attention against historical KV.
- [x] Append exactly new KV state.
- [x] Produce next logits.
- [x] Add multi-step golden test.

## 37. Full-Recompute Regression Guard

- [x] Instrument processed-token count.
- [x] Instrument KV length.
- [x] Assert decode step is incremental.
- [x] Fail if complete history is recomputed as mandatory path.
- [x] Add explicit regression test.

## 38. RoPE Incremental Regression Guard

- [x] Assert non-zero positions.
- [x] Assert position increases.
- [x] Freeze at least one non-zero-position output vector.

## 39. Sampling

- [x] Implement/use greedy sampling through Sampling contract.
- [x] Validate vocabulary length.
- [x] Select deterministic token.
- [x] Add golden tests.

## 40. Generation State Machine

- [x] Add prefill state.
- [x] Add decode state.
- [x] Add stop state.
- [x] Add cancelled state.
- [x] Add failed state.
- [x] Add state transition tests.

## 41. Generation Loop

- [x] Prefill once.
- [x] Sample token.
- [x] Decode incrementally.
- [x] Append generated token.
- [x] Repeat until stop.
- [x] Stream output.
- [x] Add deterministic sequence test.

## 42. Logits Provenance

- [x] Tag/observe model execution producing logits.
- [x] Prevent external callback provenance in mandatory path.
- [x] Add structural assertion.

## 43. RuntimeInferenceApi Cutover

- [x] Load/access Model Instance.
- [x] Create Session.
- [x] Generate.
- [x] Stream.
- [x] Cancel.
- [x] Close Session.
- [x] Release Model.
- [x] Add API tests.

## 44. Remove Caller Forward Authority

- [x] Remove `next_logits` from normal API.
- [x] Remove equivalent caller callbacks.
- [x] Isolate test-only helpers.
- [x] Mark deprecated migration APIs.
- [x] Add compile/API boundary tests.

## 45. CLI Cutover

- [x] Implement `magnetar run`.
- [x] Pass text prompt to RuntimeInferenceApi.
- [x] Render output.
- [x] Handle structured errors.
- [x] Support cancellation where existing CLI permits.
- [x] Add CLI tests.

## 46. Remove CLI Placeholder Logits

- [x] Remove fake logits.
- [x] Remove deterministic CLI model simulation.
- [x] Assert CLI cannot generate without Runtime model execution.

## 47. E2E Structural Evidence

- [x] Record Component load.
- [x] Record Model Artifact load.
- [x] Record graph fingerprint.
- [x] Record Plan generation.
- [x] Record Registry resolutions.
- [x] Record Provider.
- [x] Record Kernel IDs.
- [x] Record prefill token count.
- [x] Record KV length.
- [x] Record decode steps.
- [x] Record generated token IDs safely.

## 48. Full Native E2E

- [x] Start from prompt text.
- [x] Use tokenizer.
- [x] Use RuntimeInferenceApi.
- [x] Load real fixture.
- [x] Load Qwen WASM Component.
- [x] Execute graph.
- [x] Dispatch through Registry.
- [x] Execute Reference CPU Provider.
- [x] Use real KV.
- [x] Decode incrementally.
- [x] Sample greedily.
- [x] Produce expected output.

## 49. E2E Bypass Guards

- [x] Fail if Qwen Component not loaded.
- [x] Fail if Registry not traversed.
- [x] Fail if Provider not traversed.
- [x] Fail if Candle model execution occurs.
- [x] Fail if caller callback supplies logits.
- [x] Fail if KV is not reused.
- [x] Fail if direct Reference CPU Kernel path is used by E2E harness.

## 50. Golden Conformance

- [x] Freeze tokenization golden.
- [x] Freeze selected Kernel goldens.
- [x] Freeze prefill logits.
- [x] Freeze decode logits.
- [x] Freeze generated token sequence.
- [x] Version all golden evidence.

## 51. Failure Conformance

- [x] Invalid Tensor.
- [x] Allocation failure.
- [x] Missing Kernel.
- [x] Provider unavailable.
- [x] Invalid model config.
- [x] Missing weight.
- [x] Wrong weight shape.
- [x] Invalid Component.
- [x] KV overflow.
- [x] Invalid decode position.
- [x] Invalid token.
- [x] Cancellation.

## 52. Provider Conformance Gate

- [x] Run required Reference CPU Provider conformance profile.
- [x] Validate Kernel preparation.
- [x] Validate Kernel execution.
- [x] Validate structured failures.
- [x] Validate synchronous CompletionToken.

## 53. Runtime Conformance Gate

- [x] Validate Model Loading.
- [x] Validate Session.
- [x] Validate Generation.
- [x] Validate KV.
- [x] Validate Prepared Plan.
- [x] Validate RuntimeInferenceApi ownership.

## 54. Component Conformance Gate

- [x] Validate WASM instantiation.
- [x] Validate no ambient authority.
- [x] Validate graph semantics.
- [x] Validate no Provider/Device targeting.

## 55. Minimal Feature CI

- [x] Define first-profile feature set.
- [x] Disable Candle model execution.
- [x] Disable accelerator Providers.
- [x] Disable multi-Device if possible.
- [x] Disable generated Kernel stack if possible.
- [x] Ensure E2E still passes.

## 56. Standard CI

- [x] rustfmt.
- [x] cargo check.
- [x] clippy.
- [x] workspace unit tests.
- [x] OpenSpec validation.
- [x] WIT validation.
- [x] Provider conformance.
- [x] Runtime conformance.
- [x] Component conformance.
- [x] native E2E.
- [x] existing coverage gate.
- [x] existing dependency/security checks.

## 57. Bypass Cleanup

- [x] Remove obsolete E2E direct Kernel calls.
- [x] Remove obsolete placeholder logits.
- [x] Remove obsolete caller-forward normal API.
- [x] Remove obsolete fake KV paths.
- [x] Keep only explicitly justified test utilities.

## 58. Documentation

- [x] Document implementation phases.
- [x] Document PR dependency graph.
- [x] Document first-profile feature set.
- [x] Document E2E architecture.
- [x] Document fixture format.
- [x] Document golden evidence.
- [x] Document deferred capabilities.
- [x] Document Architecture Freeze #1.

## 59. Definition Of Done

- [x] Qwen fixture is real Model Artifact.
- [x] Qwen model architecture is WASM Component.
- [x] Runtime builds Execution Graph.
- [x] Runtime builds Prepared Execution Plan.
- [x] Registry resolves mandatory Kernels.
- [x] Reference CPU Provider executes Kernels.
- [x] Tensor memory is Memory Manager-owned.
- [x] Prefill populates KV.
- [x] Decode reuses KV.
- [x] Decode is incremental.
- [x] RoPE position is correct.
- [x] Model logits drive Sampling.
- [x] RuntimeInferenceApi owns inference.
- [x] CLI is Runtime client only.
- [x] Candle absent from native profile model execution.
- [x] E2E has no direct Kernel bypass.
- [x] Structural evidence proves path.
- [x] Minimal CI passes.

## 60. Stabilization Cut

- [x] Record baseline commit.
- [x] Record fixture/golden versions.
- [x] Record conformance evidence.
- [x] Record known deferred work.
- [x] Declare First Native Implementation Baseline.
- [x] Close Architecture Freeze #1 milestone.
