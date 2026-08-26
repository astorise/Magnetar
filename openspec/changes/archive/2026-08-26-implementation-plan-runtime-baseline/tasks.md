# Tasks

## 1. Implementation Plan Scope

- [x] Define baseline implementation plan.
- [x] Define implementation order.
- [x] Define PR sequence.
- [x] Define no-shortcut rule.
- [x] Define acceptance criteria.
- [x] Define deferred work.
- [x] Define CI gates.

## 2. PR 1 Runtime Skeleton

- [x] Create or validate module tree.
- [x] Add crate façade re-exports.
- [x] Add ID type skeletons.
- [x] Add error skeletons.
- [x] Add policy placeholder types.
- [x] Add observability event skeletons.
- [x] Add feature flag layout.
- [x] Add test fixture conventions.
- [x] Add public API no-raw-handle checks.
- [x] Run compile checks.

## 3. PR 2 Tensor And Memory

- [x] Implement TensorDescriptor.
- [x] Implement TensorResourceId.
- [x] Implement TensorResource metadata.
- [x] Implement TensorLayout.
- [x] Implement TensorDType.
- [x] Implement TensorShape.
- [x] Implement Tensor readiness.
- [x] Implement Tensor lifecycle.
- [x] Implement host memory class.
- [x] Implement contiguous layout.
- [x] Implement basic Memory Manager tracking.
- [x] Implement size accounting.
- [x] Add no raw pointer public API tests.
- [x] Add tensor and memory tests.

## 4. PR 3 Operator Catalog

- [x] Implement OperatorId.
- [x] Implement Operator metadata.
- [x] Implement required-now classification.
- [x] Implement placeholder classification.
- [x] Implement unsupported classification.
- [x] Implement shape validation helpers.
- [x] Implement dtype validation helpers.
- [x] Implement layout validation helpers.
- [x] Add operator conformance fixtures.
- [x] Add first scope tests.

## 5. PR 4 Reference CPU Provider

- [x] Implement Reference CPU Provider identity.
- [x] Implement CPU Device metadata.
- [x] Implement Provider status snapshot.
- [x] Implement kernel advertisements.
- [x] Implement host contiguous f32 execution path.
- [x] Implement embedding kernel.
- [x] Implement matmul kernel.
- [x] Implement RMSNorm kernel.
- [x] Implement RoPE kernel.
- [x] Implement attention kernel.
- [x] Implement softmax kernel.
- [x] Implement SiLU kernel.
- [x] Implement add kernel.
- [x] Implement mul kernel.
- [x] Implement residual-add kernel.
- [x] Add structured CPU errors.
- [x] Add CPU conformance fixtures.

## 6. PR 5 Kernel Registry And Dispatch

- [x] Implement KernelAdvertisement validation.
- [x] Implement KernelCandidate.
- [x] Implement KernelSelectionRequest.
- [x] Implement KernelSelectionResult.
- [x] Implement KernelDispatchPlan.
- [x] Implement KernelDispatchResult.
- [x] Implement candidate filtering.
- [x] Implement Resource Affinity validation.
- [x] Implement Memory Manager feasibility checks.
- [x] Implement Provider readiness checks.
- [x] Implement Device readiness checks.
- [x] Implement dispatch revalidation.
- [x] Implement explicit fallback policy placeholder.
- [x] Add no direct Provider execution E2E guard.

## 7. PR 6 Model Artifact Loading Instance

- [x] Implement fixture Model Artifact metadata.
- [x] Implement artifact manifest validation.
- [x] Implement artifact trust validation.
- [x] Implement tensor inventory validation hooks.
- [x] Implement Model Loading request.
- [x] Implement Model Loading lifecycle.
- [x] Implement ModelInstanceId.
- [x] Implement Model Instance readiness.
- [x] Implement unload cleanup path.
- [x] Add loading and instance tests.

## 8. PR 7 Tokenizer Fixture

- [x] Implement tokenizer fixture.
- [x] Implement encode path.
- [x] Implement decode path.
- [x] Implement streaming decode path.
- [x] Implement special token metadata.
- [x] Implement tokenizer/model compatibility validation.
- [x] Add raw prompt redaction tests.
- [x] Add tokenizer tests.

## 9. PR 8 Qwen Baseline Component

- [x] Implement Qwen config validation.
- [x] Implement Qwen tensor inventory validation.
- [x] Implement Qwen target modules.
- [x] Implement Qwen KV cache metadata.
- [x] Implement Qwen tokenizer compatibility metadata.
- [x] Implement prefill graph production.
- [x] Implement decode graph production.
- [x] Implement required operator scope validation.
- [x] Add no QwenProvider test.
- [x] Add no direct Kernel/Provider access test.
- [x] Add Qwen baseline conformance fixtures.

## 10. PR 9 Generation And Sampling

- [x] Implement GenerationRequest.
- [x] Implement generation lifecycle.
- [x] Implement prefill orchestration.
- [x] Implement decode orchestration.
- [x] Implement greedy sampling.
- [x] Implement stop conditions.
- [x] Implement max new tokens.
- [x] Implement usage accounting.
- [x] Implement cancellation points.
- [x] Implement streaming event skeleton.
- [x] Add generation and sampling tests.

## 11. PR 10 Runtime Inference API

- [x] Implement model resolution API.
- [x] Implement model loading API.
- [x] Implement session API.
- [x] Implement tokenization API.
- [x] Implement generation API.
- [x] Implement streaming API.
- [x] Implement cancellation API.
- [x] Implement diagnostics API.
- [x] Implement usage reporting.
- [x] Add handle redaction tests.
- [x] Add inference-only scope tests.

## 12. PR 11 CLI Boundary Harness

- [x] Add minimal CLI boundary harness or tests.
- [x] Verify CLI sends explicit prompt/context.
- [x] Verify Runtime does not read workspace files.
- [x] Verify Runtime does not execute tools.
- [x] Verify Runtime does not execute shell/process.
- [x] Verify Runtime does not execute Git.
- [x] Verify Runtime receives no ambient CLI authority.
- [x] Verify Runtime structured errors are preserved.
- [x] Add CLI boundary tests.

## 13. PR 12 E2E Local Conformance

- [x] Implement fixture model.
- [x] Implement fixture tokenizer.
- [x] Implement fixture artifact.
- [x] Implement Reference CPU path.
- [x] Implement Qwen baseline graph path.
- [x] Use Runtime Inference API entrypoint.
- [x] Validate session lifecycle.
- [x] Validate generation.
- [x] Validate streaming.
- [x] Validate diagnostics.
- [x] Validate redaction.
- [x] Validate failure cases.
- [x] Generate machine-readable report.
- [x] Integrate into CI.

## 14. No Shortcut Checks

- [x] Check no direct Provider invocation in E2E success path.
- [x] Check no direct Kernel invocation from Model Component.
- [x] Check no Model Artifact validation bypass.
- [x] Check no Model Loading bypass.
- [x] Check no Model Instance bypass.
- [x] Check no Tokenizer bypass for text prompt.
- [x] Check no Kernel Registry bypass.
- [x] Check no Memory Manager bypass.
- [x] Check no Runtime Inference API bypass for E2E.
- [x] Check no raw tensor pointer exposure.
- [x] Check no raw Provider/Device/Kernel handle exposure.
- [x] Check no silent dtype conversion.
- [x] Check no silent layout conversion.
- [x] Check no silent CPU fallback.
- [x] Check no Runtime filesystem access.
- [x] Check no Runtime tool execution.
- [x] Check no Runtime shell/process execution.
- [x] Check no Runtime Git execution.

## 15. Acceptance Gates

- [x] Verify all modules compile.
- [x] Verify all public IDs are opaque.
- [x] Verify Tensor Resource has no raw pointer public API.
- [x] Verify Memory Manager tracks host contiguous tensors.
- [x] Verify required-now Operators are classified.
- [x] Verify Reference CPU Provider advertises required kernels.
- [x] Verify Kernel Registry selects Reference CPU kernels.
- [x] Verify Model Loading validates fixture artifact.
- [x] Verify Qwen baseline produces validated graphs.
- [x] Verify Generation produces deterministic output.
- [x] Verify Runtime Inference API exposes baseline inference.
- [x] Verify CLI boundary tests pass.
- [x] Verify E2E local conformance passes CPU-only.
- [x] Verify diagnostics and observability are redacted.
- [x] Verify OpenSpec validation passes.
- [x] Verify coverage gate passes.

## 16. Deferred Work Tracking

- [x] Track production model download UX as deferred.
- [x] Track large Qwen model execution as deferred.
- [x] Track optimized CPU kernels as deferred.
- [x] Track SIMD/BLAS acceleration as deferred.
- [x] Track CUDA Provider as deferred.
- [x] Track Metal Provider as deferred.
- [x] Track OpenVINO Provider as deferred.
- [x] Track QNN Provider as deferred.
- [x] Track WebGPU Provider as deferred.
- [x] Track GGUF support as deferred.
- [x] Track full quantized inference as deferred.
- [x] Track paged attention as deferred.
- [x] Track flash attention as deferred.
- [x] Track speculative decoding as deferred.
- [x] Track beam search as deferred.
- [x] Track agent/tool runtime as deferred.
- [x] Track production CLI UX as deferred.
- [x] Track HTTP server API as deferred.
- [x] Track Tachyon distributed conformance as deferred.

## 17. CI Gates

- [x] Run cargo fmt.
- [x] Run cargo check.
- [x] Run cargo clippy.
- [x] Run cargo test.
- [x] Run wasm32 check where feasible.
- [x] Run OpenSpec validation.
- [x] Run unit tests.
- [x] Run contract tests.
- [x] Run Reference CPU conformance.
- [x] Run first operator scope conformance.
- [x] Run Qwen baseline conformance.
- [x] Run Runtime Inference API tests.
- [x] Run CLI boundary tests.
- [x] Run E2E local inference conformance.
- [x] Run coverage validation.
- [x] Ensure GPU checks are not required.

## 18. Observability Tests

- [x] Test model loading observations.
- [x] Test Model Instance readiness observations.
- [x] Test session lifecycle observations.
- [x] Test tokenization observations.
- [x] Test generation observations.
- [x] Test operator planning observations.
- [x] Test Kernel Registry selection observations.
- [x] Test Reference CPU dispatch observations.
- [x] Test memory allocation observations.
- [x] Test streaming observations.
- [x] Test cancellation observations.
- [x] Test error observations.
- [x] Test E2E report observations.
- [x] Verify redaction by default.

## 19. Documentation

- [x] Document implementation plan.
- [x] Document PR sequence.
- [x] Document module order.
- [x] Document no-shortcut rule.
- [x] Document acceptance gates.
- [x] Document deferred work.
- [x] Document CI gates.
- [x] Document implementation baseline.
- [x] Document post-baseline roadmap.

## 20. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run full test suite.
- [x] Run conformance suites.
- [x] Run E2E local conformance.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify implementation can begin from PR 1.
- [x] Verify no architecture bypass remains unspecified.