# Tasks

## 1. Roadmap Scope

- [x] Define post-baseline Provider roadmap.
- [x] Document Reference CPU as correctness baseline.
- [x] Document optimized Provider principles.
- [x] Document no model-family Provider rule.
- [x] Document conformance-first Provider admission.
- [x] Document non-goals.

## 2. Provider Roadmap Module

- [x] Create provider roadmap documentation or module.
- [x] Define roadmap phases.
- [x] Define Provider feature metadata.
- [x] Define Provider readiness gates.
- [x] Define Provider conformance gates.
- [x] Add roadmap validation tests where applicable.

## 3. Optimized CPU Provider Phase

- [x] Define optimized CPU Provider scope.
- [x] Allow SIMD as optional.
- [x] Allow BLAS as optional.
- [x] Allow thread pool execution as optional.
- [x] Allow cache-aware kernels as optional.
- [x] Allow fused kernels as optional.
- [x] Preserve Reference CPU correctness baseline.
- [x] Add optimized CPU conformance requirements.

## 4. CUDA Provider Phase

- [x] Define CUDA Provider scope.
- [x] Define CUDA Device metadata.
- [x] Define CUDA memory classes.
- [x] Define CUDA kernel advertisement expectations.
- [x] Define CUDA stream/internal handle boundary.
- [x] Define CUDA fallback policy.
- [x] Define CUDA conformance profile.
- [x] Add CUDA feature placeholders.

## 5. Metal Provider Phase

- [x] Define Metal Provider scope.
- [x] Define Metal Device metadata.
- [x] Define Metal buffer/internal handle boundary.
- [x] Define Metal command queue/internal handle boundary.
- [x] Define Metal pipeline/internal handle boundary.
- [x] Define Metal fallback policy.
- [x] Define Metal conformance profile.
- [x] Add Metal feature placeholders.

## 6. OpenVINO Provider Phase

- [x] Define OpenVINO Provider scope.
- [x] Define compiled graph opaque boundary.
- [x] Define static shape profile support.
- [x] Define dynamic shape support reporting.
- [x] Define quantized support reporting.
- [x] Define OpenVINO fallback policy.
- [x] Define OpenVINO conformance profile.
- [x] Add OpenVINO feature placeholders.

## 7. QNN Provider Phase

- [x] Define QNN Provider scope.
- [x] Define mobile/NPU execution scope.
- [x] Define static shape constraints.
- [x] Define quantized execution support reporting.
- [x] Define native QNN handle boundary.
- [x] Define QNN fallback policy.
- [x] Define QNN conformance profile.
- [x] Add QNN feature placeholders.

## 8. WebGPU Provider Phase

- [x] Define WebGPU Provider scope.
- [x] Define browser-compatible memory.
- [x] Define WebGPU buffer layout metadata.
- [x] Define WGSL kernel placeholder.
- [x] Define reduced dtype support reporting.
- [x] Define browser constraint reporting.
- [x] Define WebGPU conformance profile.
- [x] Add WebGPU feature placeholders.

## 9. Kernel Fusion

- [x] Define fused kernel metadata.
- [x] Define semantic equivalence declaration.
- [x] Define graph fragment matching.
- [x] Define precision tolerance.
- [x] Define fallback behavior.
- [x] Reject invalid fusion.
- [x] Add fusion conformance requirements.

## 10. Advanced Attention

- [x] Define flash attention roadmap.
- [x] Define paged attention roadmap.
- [x] Define sliding window attention roadmap.
- [x] Define block sparse attention roadmap.
- [x] Define GQA/MQA optimized attention roadmap.
- [x] Define KV-cache-aware attention roadmap.
- [x] Define advanced attention metadata.
- [x] Define advanced attention conformance gates.

## 11. Quantized Execution

- [x] Define quantization method metadata.
- [x] Define storage dtype metadata.
- [x] Define compute dtype metadata.
- [x] Define accumulation dtype metadata.
- [x] Define scale metadata.
- [x] Define zero-point metadata.
- [x] Define group size metadata.
- [x] Define packing layout metadata.
- [x] Define dequantization behavior metadata.
- [x] Define quantized Operator support metadata.
- [x] Define quantized conformance tolerance.
- [x] Prevent hidden quantization/dequantization.

## 12. Layout Expansion

- [x] Define blocked layout post-baseline support.
- [x] Define paged layout post-baseline support.
- [x] Define packed-quantized layout support.
- [x] Define attention-specific layout support.
- [x] Define provider-owned opaque layout support.
- [x] Define WebGPU buffer layout support.
- [x] Require explicit layout conversion.
- [x] Add layout expansion tests.

## 13. Memory Expansion

- [x] Define device memory post-baseline support.
- [x] Define pinned-host memory support.
- [x] Define unified memory support.
- [x] Define shared memory support.
- [x] Define provider-owned memory support.
- [x] Define browser-linear-memory support.
- [x] Define future-webgpu-buffer support.
- [x] Require Memory Manager tracking.
- [x] Add memory expansion tests.

## 14. Provider Conformance Profiles

- [x] Define provider-core profile.
- [x] Define provider-compute profile.
- [x] Define provider-data-movement profile.
- [x] Define provider-cancellation profile.
- [x] Define provider-observability profile.
- [x] Define provider-dynamic-abi profile.
- [x] Define provider-quantized profile.
- [x] Define provider-advanced-attention profile.
- [x] Define provider-fused-kernel profile.
- [x] Define provider-browser profile.
- [x] Add profile reporting.

## 15. Benchmarks

- [x] Define benchmark separation from conformance.
- [x] Define prefill latency benchmark placeholder.
- [x] Define decode latency benchmark placeholder.
- [x] Define tokens per second benchmark placeholder.
- [x] Define memory footprint benchmark placeholder.
- [x] Define batching throughput benchmark placeholder.
- [x] Define cache hit behavior benchmark placeholder.
- [x] Define transfer overhead benchmark placeholder.
- [x] Define kernel dispatch overhead benchmark placeholder.
- [x] Ensure benchmarks do not replace correctness tests.

## 16. Fallback Policy

- [x] Define optimized Provider to Reference CPU fallback.
- [x] Define CUDA to CPU fallback.
- [x] Define Metal to CPU fallback.
- [x] Define OpenVINO to CPU fallback.
- [x] Define QNN to CPU fallback.
- [x] Define WebGPU fallback.
- [x] Prevent silent fallback.
- [x] Validate Resource Affinity before fallback.
- [x] Validate dtype/layout before fallback.
- [x] Validate memory policy before fallback.
- [x] Validate privacy and precision policy before fallback.

## 17. Runtime API Stability

- [x] Ensure basic Runtime Inference API remains Provider-independent.
- [x] Add Provider policy preference placeholders.
- [x] Add redacted Provider diagnostics support.
- [x] Prevent Provider-specific handles in API.
- [x] Add Runtime API stability tests.

## 18. CLI Boundary Stability

- [x] Allow CLI to display redacted Provider diagnostics.
- [x] Allow CLI to pass policy preferences.
- [x] Prevent CLI from selecting raw Provider handles.
- [x] Preserve Runtime-owned Provider selection.
- [x] Add CLI Provider boundary tests.

## 19. Error Model

- [x] Define provider-roadmap-unsupported error.
- [x] Define optimized-cpu-provider-unavailable error.
- [x] Define cuda-provider-unavailable error.
- [x] Define metal-provider-unavailable error.
- [x] Define openvino-provider-unavailable error.
- [x] Define qnn-provider-unavailable error.
- [x] Define webgpu-provider-unavailable error.
- [x] Define provider-feature-unsupported error.
- [x] Define provider-layout-unsupported error.
- [x] Define provider-dtype-unsupported error.
- [x] Define provider-memory-class-unsupported error.
- [x] Define provider-advanced-attention-unsupported error.
- [x] Define provider-quantization-unsupported error.
- [x] Define provider-fusion-invalid error.
- [x] Define provider-conformance-failed error.
- [x] Define provider-benchmark-failed error.
- [x] Define provider-fallback-denied error.
- [x] Define provider-native-handle-exposure-denied error.
- [x] Define internal-provider-roadmap error.

## 20. Observability

- [x] Emit Provider roadmap feature discovered observation.
- [x] Emit Provider capability advertised observation.
- [x] Emit Provider capability rejected observation.
- [x] Emit optimized Provider selected observation.
- [x] Emit advanced attention selected observation.
- [x] Emit quantized kernel selected observation.
- [x] Emit fused kernel selected observation.
- [x] Emit fallback considered observation.
- [x] Emit fallback used observation.
- [x] Emit fallback denied observation.
- [x] Emit Provider conformance passed observation.
- [x] Emit Provider conformance failed observation.
- [x] Emit benchmark executed observation.
- [x] Emit benchmark skipped observation.
- [x] Verify default redaction.

## 21. Tests

- [x] Test Reference CPU remains correctness baseline.
- [x] Test optimized Provider does not redefine Operator semantics.
- [x] Test model-family Provider names are rejected.
- [x] Test native handle exposure is denied.
- [x] Test fused kernel requires semantic declaration.
- [x] Test quantized path requires explicit metadata.
- [x] Test advanced attention unsupported path fails explicitly.
- [x] Test fallback denied by default.
- [x] Test Runtime API remains Provider-independent.
- [x] Test CLI receives redacted Provider diagnostics only.

## 22. Documentation

- [x] Document post-baseline Provider roadmap.
- [x] Document optimized CPU phase.
- [x] Document CUDA phase.
- [x] Document Metal phase.
- [x] Document OpenVINO phase.
- [x] Document QNN phase.
- [x] Document WebGPU phase.
- [x] Document advanced attention roadmap.
- [x] Document quantization roadmap.
- [x] Document layout expansion.
- [x] Document memory expansion.
- [x] Document conformance profiles.
- [x] Document benchmarks.
- [x] Document fallback policy.
- [x] Document non-goals.

## 23. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify roadmap does not alter baseline acceptance criteria.
- [x] Verify no Provider bypass is introduced.
- [x] Verify no model-family Provider is introduced.
- [x] Verify Provider-specific features remain behind contracts.
