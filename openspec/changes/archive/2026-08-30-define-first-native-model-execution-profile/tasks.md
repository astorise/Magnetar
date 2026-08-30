# Tasks

## 1. Profile Definition

- [x] Define FirstNativeModelExecutionProfile.
- [x] Define mandatory capabilities.
- [x] Define deferred capabilities.
- [x] Define profile version.
- [x] Add profile validation tests.

## 2. Native Execution Definition

- [x] Require Magnetar Operator execution.
- [x] Require Magnetar Kernel execution.
- [x] Require Kernel Registry.
- [x] Require Reference CPU Provider.
- [x] Exclude Candle from mandatory execution path.
- [x] Add structural conformance checks.

## 3. Single Host Baseline

- [x] Require one local Runtime.
- [x] Require one Provider.
- [x] Require one logical CPU Device.
- [x] Keep multi-Device optional.
- [x] Add baseline topology tests.

## 4. Component Engine Baseline

- [x] Require Platform Component Engine.
- [x] Require native Wasmtime implementation.
- [x] Keep Wasmtime private.
- [x] Disable ambient WASI authority.
- [x] Add Component Engine tests.

## 5. Qwen Model Component

- [x] Build Qwen architecture as WASM Component.
- [x] Use existing Qwen Model Component contract.
- [x] Consume normalized model configuration.
- [x] Consume logical model Resources.
- [x] Emit/use portable Operator graph.
- [x] Prevent Provider/Device selection.
- [x] Add Component tests.

## 6. Tiny Qwen Fixture

- [x] Define fixture version.
- [x] Fix vocabulary size.
- [x] Fix hidden size.
- [x] Fix layer count.
- [x] Fix attention head count.
- [x] Fix KV head count.
- [x] Fix head dimension.
- [x] Fix intermediate size.
- [x] Fix maximum context.
- [x] Fix dtype to f32.
- [x] Add fixture metadata validation.

## 7. Fixture Weights

- [x] Define deterministic weight generation or fixed artifact.
- [x] Persist fixture weights.
- [x] Record fixture digest.
- [x] Add integrity test.
- [x] Prevent test-only Kernel access to weight-generation knowledge.

## 8. Model Artifact Fixture

- [x] Provide model configuration.
- [x] Provide weights.
- [x] Provide tokenizer artifact.
- [x] Provide model metadata.
- [x] Normalize through Model Loading contract.
- [x] Add missing-artifact failure tests.

## 9. First Physical Model Format

- [x] Support minimal fixture model package.
- [x] Prefer single safetensors weight file where practical.
- [x] Support minimal Qwen configuration subset.
- [x] Defer sharded models.
- [x] Defer GGUF.
- [x] Defer quantized formats.

## 10. Tokenizer

- [x] Use Tokenizer contract.
- [x] Define deterministic fixture tokenizer.
- [x] Support encode.
- [x] Support decode.
- [x] Define special tokens needed by fixture.
- [x] Add tokenizer golden tests.

## 11. Required Operator Catalog

- [x] Implement embedding.
- [x] Implement matmul.
- [x] Implement rmsnorm.
- [x] Implement rope.
- [x] Implement attention.
- [x] Implement softmax.
- [x] Implement silu.
- [x] Implement add.
- [x] Implement mul.
- [x] Implement residual-add.
- [x] Implement dtype-conversion.
- [x] Implement layout-conversion.

## 12. Operator Validation

- [x] Validate Tensor ranks.
- [x] Validate shapes.
- [x] Validate dtype.
- [x] Validate layout.
- [x] Validate attributes.
- [x] Add negative Operator tests.

## 13. Reference CPU Provider

- [x] Implement all mandatory Kernel capabilities.
- [x] Support one logical CPU Device.
- [x] Support f32.
- [x] Support normal Kernel preparation.
- [x] Support synchronous ExecutionStream baseline.
- [x] Add Provider conformance tests.

## 14. Reference CPU Kernels

- [x] Implement correctness-first embedding.
- [x] Implement correctness-first matmul.
- [x] Implement correctness-first rmsnorm.
- [x] Implement correctness-first rope.
- [x] Implement correctness-first attention.
- [x] Implement correctness-first softmax.
- [x] Implement correctness-first silu.
- [x] Implement correctness-first add/mul/residual.
- [x] Implement conversion Kernels.
- [x] Add mathematical golden tests.

## 15. Numerical Semantics

- [x] Define/reference f32 tolerances.
- [x] Test zero-sized/invalid cases where contract permits.
- [x] Test NaN/Inf behavior.
- [x] Test softmax stability.
- [x] Test RMSNorm epsilon behavior.
- [x] Add deterministic CPU tests.

## 16. RoPE Incremental Correctness

- [x] Support non-zero position.
- [x] Support per-token decode position.
- [x] Add position-0 test.
- [x] Add non-zero-position tests.
- [x] Add multi-token progression test.

## 17. Attention Correctness

- [x] Support Qwen fixture head geometry.
- [x] Support KV head mapping.
- [x] Support causal masking.
- [x] Consume prior KV during decode.
- [x] Add prefill Attention test.
- [x] Add incremental Attention tests.

## 18. Kernel Registry

- [x] Register Reference CPU Kernels.
- [x] Validate registration errors.
- [x] Expose candidate lookup.
- [x] Apply eligibility.
- [x] Apply deterministic selection.
- [x] Add Registry tests.

## 19. No Registry Bypass

- [x] Remove/directly forbid E2E Reference CPU calls.
- [x] Add test that records Registry resolution.
- [x] Add test that records Provider execution.
- [x] Add architecture guard where practical.

## 20. Prepared Kernel

- [x] Prepare selected Reference CPU Kernel.
- [x] Produce opaque PreparedKernelId.
- [x] Keep native/function implementation private to Provider.
- [x] Add preparation tests.

## 21. Execution Graph

- [x] Build actual graph from Qwen Component.
- [x] Validate graph semantics.
- [x] Bind model Resources.
- [x] Preserve Operator ordering.
- [x] Add graph fingerprint.
- [x] Add graph tests.

## 22. No Monolithic Qwen Bypass

- [x] Prevent dedicated Qwen-forward backend path.
- [x] Require Qwen graph through portable Operators.
- [x] Add structural test.

## 23. Prepared Execution Plan

- [x] Build Plan for Qwen fixture.
- [x] Bind Kernels.
- [x] Bind CPU Device.
- [x] Bind Resource slots.
- [x] Bind execution order.
- [x] Bind AllocationPlan.
- [x] Mark Plan ready.
- [x] Add Plan tests.

## 24. Decode Plan Reuse

- [x] Cache/reuse compatible decode Plan.
- [x] Avoid full Registry resolution every token.
- [x] Add Plan reuse evidence.
- [x] Add invalid Plan failure test.

## 25. Execution Stream

- [x] Implement synchronous CPU ExecutionStream.
- [x] Submit Prepared Kernel.
- [x] Produce completed CompletionToken.
- [x] Preserve dependency semantics.
- [x] Add stream tests.

## 26. Resource Readiness

- [x] Mark Kernel outputs ready.
- [x] Track dependencies.
- [x] Protect reuse.
- [x] Add synchronous readiness tests.

## 27. Tensor Resource

- [x] Route all model tensors through TensorResource.
- [x] Preserve descriptor semantics.
- [x] Support contiguous dense baseline.
- [x] Support minimal Views required.
- [x] Add Tensor lifecycle tests.

## 28. Memory Manager

- [x] Own all Tensor allocations.
- [x] Implement simple pool/arena.
- [x] Enforce bounds.
- [x] Enforce alignment.
- [x] Track lifetime.
- [x] Track aliases.
- [x] Add Memory Manager tests.

## 29. Allocation Plan

- [x] Build conservative AllocationPlan.
- [x] Bind persistent slots.
- [x] Bind KV storage.
- [x] Bind transient/workspace slots.
- [x] Support safe reuse.
- [x] Add allocation tests.

## 30. No Mandatory Per Operator Native Allocation

- [x] Reuse planned workspace.
- [x] Reuse transient storage when safe.
- [x] Add instrumentation for allocation count.
- [x] Add decode reuse test.

## 31. Model Loading

- [x] Parse/normalize fixture configuration.
- [x] Load weights.
- [x] Validate required tensors.
- [x] Validate tensor shapes.
- [x] Instantiate Qwen Model Component.
- [x] Build Model Instance.
- [x] Add loading failures.

## 32. Model Instance

- [x] Bind Model Artifact.
- [x] Bind Qwen Model Component.
- [x] Bind Reference CPU Provider/Device.
- [x] Bind Prepared Plans.
- [x] Expose readiness.
- [x] Add lifecycle tests.

## 33. Inference Session

- [x] Create Session.
- [x] Bind Model Instance.
- [x] Own/reference KV state.
- [x] Track token position.
- [x] Track cancellation.
- [x] Add Session tests.

## 34. KV Cache

- [x] Allocate per-layer K state.
- [x] Allocate per-layer V state.
- [x] Track sequence length.
- [x] Support append.
- [x] Support read.
- [x] Validate bounds.
- [x] Add KV tests.

## 35. Prefill

- [x] Tokenize prompt.
- [x] Execute prompt through Qwen graph.
- [x] Populate KV cache.
- [x] Produce final prefill logits.
- [x] Add prefill golden test.

## 36. Incremental Decode

- [x] Consume one/new token at decode step.
- [x] Reuse existing KV.
- [x] Append only new K/V.
- [x] Use correct position.
- [x] Produce next logits.
- [x] Add multi-step decode test.

## 37. Prevent Full Sequence Recompute

- [x] Instrument token counts or graph inputs.
- [x] Prove decode does not execute complete token history as mandatory path.
- [x] Add regression test.

## 38. Sampling

- [x] Support greedy sampling.
- [x] Use Sampling contract.
- [x] Validate logits length.
- [x] Add greedy golden tests.
- [x] Keep stochastic sampling optional.

## 39. Generation Loop

- [x] Prefill prompt.
- [x] Sample first generated token.
- [x] Run incremental decode.
- [x] Append token.
- [x] Apply stop criteria.
- [x] Decode output.
- [x] Support streaming callbacks/events through Runtime API.
- [x] Add generation tests.

## 40. RuntimeInferenceApi

- [x] Load/open Model Instance through Runtime.
- [x] Create generation/session.
- [x] Execute generation.
- [x] Stream output.
- [x] Cancel.
- [x] Release Session/Model.
- [x] Add API tests.

## 41. Remove Caller Logits Authority

- [x] Remove `next_logits` from ordinary inference execution API.
- [x] Remove equivalent caller forward callbacks.
- [x] Isolate test hooks.
- [x] Add API boundary test.

## 42. CLI Boundary

- [x] Implement `magnetar run` path.
- [x] Pass prompt text to RuntimeInferenceApi.
- [x] Receive generated output.
- [x] Do not call Provider.
- [x] Do not call Kernel Registry.
- [x] Do not compute logits.
- [x] Do not own KV.
- [x] Add CLI boundary tests.

## 43. Mandatory E2E Fixture

- [x] Start from text.
- [x] Resolve fixture model.
- [x] Load WASM Qwen Component.
- [x] Load Model Artifact.
- [x] Tokenize.
- [x] Prefill.
- [x] Incrementally decode.
- [x] Greedy sample.
- [x] Produce deterministic output.

## 44. E2E Structural Evidence

- [x] Observe Model Component load.
- [x] Observe graph build.
- [x] Observe Plan build.
- [x] Observe Registry lookup.
- [x] Observe Reference CPU Provider selection.
- [x] Observe Kernel execution.
- [x] Observe KV append.
- [x] Observe incremental decode.
- [x] Observe generated token.
- [x] Redact sensitive data.

## 45. Golden Evidence

- [x] Freeze tokenizer golden.
- [x] Freeze selected Kernel golden vectors.
- [x] Freeze prefill logits.
- [x] Freeze at least one decode-step logits vector.
- [x] Freeze expected greedy token sequence.
- [x] Version golden evidence.

## 46. Failure Conformance

- [x] Invalid model config.
- [x] Missing weight.
- [x] Wrong weight shape.
- [x] Kernel unavailable.
- [x] Invalid graph.
- [x] Invalid Plan.
- [x] KV overflow/position mismatch.
- [x] Invalid token.
- [x] Component load failure.
- [x] Cancellation.

## 47. Security Boundary

- [x] No ambient Component network.
- [x] No ambient Component filesystem.
- [x] No native Tensor pointers to Component.
- [x] No Provider handles to Component.
- [x] No secrets in Component.
- [x] Add authority tests.

## 48. Observability

- [x] Trace safe E2E stages.
- [x] Trace Kernel IDs.
- [x] Trace Provider.
- [x] Trace Plan generation.
- [x] Trace KV progression without contents.
- [x] Trace generation completion.
- [x] Redact prompts by default where required.
- [x] Redact model/Tensor contents.

## 49. Deferred Multi Device

- [x] Mark multi-Device optional for profile.
- [x] Mark Tensor Parallel deferred.
- [x] Mark collectives deferred.
- [x] Ensure absence does not fail conformance.

## 50. Deferred Generated Kernels

- [x] Mark Provider compilation optional.
- [x] Mark generated Kernel qualification optional.
- [x] Mark Kernel Artifact ingestion optional.
- [x] Mark hot swap optional.

## 51. Deferred Optimization

- [x] Mark Runtime autotuning optional.
- [x] Mark adaptive feedback optional.
- [x] Mark Performance Model optional.
- [x] Ensure static Reference CPU selection remains conformant.

## 52. Deferred Accelerators

- [x] Mark CUDA optional.
- [x] Mark Metal optional.
- [x] Mark OpenVINO optional.
- [x] Mark QNN optional.
- [x] Mark WebGPU optional.

## 53. Deferred Precision/Quantization

- [x] Require f32 only.
- [x] Mark f16 optional.
- [x] Mark bf16 optional.
- [x] Mark fp8 optional.
- [x] Mark int8/int4 optional.

## 54. Deferred Advanced Memory

- [x] Mark compaction optional.
- [x] Mark overcommit optional.
- [x] Mark peer access optional.
- [x] Mark cross-Provider zero-copy optional.
- [x] Mark advanced pool classes optional.

## 55. Deferred Advanced Serving

- [x] Mark production continuous batching optimization optional.
- [x] Mark Prefix Cache optimization optional.
- [x] Mark paged KV optional.
- [x] Preserve contracts for later implementation.

## 56. Documentation

- [x] Document mandatory profile.
- [x] Document deferred features.
- [x] Document native definition.
- [x] Document Qwen fixture.
- [x] Document required Operators/Kernels.
- [x] Document real incremental KV requirement.
- [x] Document CLI-to-Kernel E2E path.
- [x] Document no-Candle conformance path.

## 57. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify profile is implementable without deferred features.
- [x] Verify no caller logits authority remains in profile.
- [x] Verify WASM Component does not select Provider/Device.
- [x] Verify E2E requires Registry/Dispatch.
- [x] Verify Reference CPU path is native Magnetar execution.
- [x] Verify real KV incremental decode is mandatory.
- [x] Verify advanced features no longer block stabilization #1.
