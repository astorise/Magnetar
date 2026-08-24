# Tasks

## 1. Scope Classification

- [x] Define required-now tier.
- [x] Define required-for-first-decoder-model tier.
- [x] Define placeholder tier.
- [x] Define explicitly-unsupported tier.
- [x] Define future-optimized tier.
- [x] Add scope classification documentation.

## 2. Operator Scope Module

- [x] Create `operator_scope` module or equivalent.
- [x] Export first implementation scope metadata.
- [x] Keep scope platform-neutral.
- [x] Keep scope independent from Provider-specific kernels.
- [x] Add module-level documentation.

## 3. Required-Now Operators

- [x] Mark embedding required-now.
- [x] Mark matmul required-now.
- [x] Mark rmsnorm required-now.
- [x] Mark rope required-now.
- [x] Mark attention required-now.
- [x] Mark softmax required-now.
- [x] Mark silu required-now.
- [x] Mark add required-now.
- [x] Mark mul required-now.
- [x] Mark residual-add required-now.
- [x] Mark dtype-conversion required-now.
- [x] Mark layout-conversion required-now.
- [x] Add required-now tests.

## 4. Placeholder Operators

- [x] Mark batched-matmul placeholder.
- [x] Mark layernorm placeholder.
- [x] Mark gelu placeholder unless implemented.
- [x] Mark dequantize placeholder unless implemented.
- [x] Mark quantize placeholder.
- [x] Mark requantize placeholder.
- [x] Mark quantized-matmul placeholder.
- [x] Mark paged-attention placeholder.
- [x] Mark sampling-helper placeholder.
- [x] Mark logits-processor-helper placeholder.
- [x] Mark layout-pack placeholder.
- [x] Mark layout-unpack placeholder.
- [x] Add placeholder tests.

## 5. Explicitly Unsupported Operators

- [x] Mark flash-attention unsupported.
- [x] Mark grouped quantization kernels unsupported.
- [x] Mark MoE dispatch unsupported.
- [x] Mark speculative decoding helpers unsupported.
- [x] Mark beam search helpers unsupported.
- [x] Mark training operators unsupported.
- [x] Mark gradient operators unsupported.
- [x] Add unsupported tests.

## 6. Future Optimized Operators

- [x] Track flash-attention as future optimized.
- [x] Track fused-rmsnorm as future optimized.
- [x] Track fused-mlp as future optimized.
- [x] Track fused-rope-attention as future optimized.
- [x] Track tensorcore-matmul as future optimized.
- [x] Track simd-rmsnorm as future optimized.
- [x] Track paged-attention as future optimized.
- [x] Track quantized-matmul as future optimized.
- [x] Add future-optimized metadata tests.

## 7. Reference CPU Coverage

- [x] Require CPU kernel for embedding.
- [x] Require CPU kernel for matmul.
- [x] Require CPU kernel for rmsnorm.
- [x] Require CPU kernel for rope.
- [x] Require CPU kernel for attention.
- [x] Require CPU kernel for softmax.
- [x] Require CPU kernel for silu.
- [x] Require CPU kernel for add.
- [x] Require CPU kernel for mul.
- [x] Require CPU kernel for residual-add.
- [x] Require CPU kernel for dtype-conversion.
- [x] Require CPU kernel for layout-conversion.
- [x] Ensure placeholders are not silently advertised.
- [x] Add CPU coverage tests.

## 8. DType Scope

- [x] Require f32 compute support.
- [x] Require f32 storage support.
- [x] Require i32 token ID support.
- [x] Require u32 token ID support where supported.
- [x] Require bool mask support where masks exist.
- [x] Mark f16 compute placeholder unless implemented.
- [x] Mark bf16 compute placeholder unless implemented.
- [x] Mark int8 quantized compute placeholder unless implemented.
- [x] Mark uint8 quantized compute placeholder unless implemented.
- [x] Prevent silent dtype conversion.
- [x] Add dtype scope tests.

## 9. Layout Scope

- [x] Require contiguous layout support.
- [x] Mark strided layout placeholder unless implemented.
- [x] Mark paged layout placeholder.
- [x] Mark blocked layout future.
- [x] Mark packed quantized layout future.
- [x] Keep Provider-owned opaque layout internal.
- [x] Prevent silent layout conversion.
- [x] Add layout scope tests.

## 10. Shape Scope

- [x] Validate tensor rank.
- [x] Validate batch dimension.
- [x] Validate sequence length.
- [x] Validate hidden size.
- [x] Validate head count.
- [x] Validate KV head count.
- [x] Validate head dimension.
- [x] Validate intermediate size.
- [x] Validate vocabulary size.
- [x] Validate matmul compatibility.
- [x] Validate broadcasting policy.
- [x] Add shape scope tests.

## 11. Attention Scope

- [x] Support causal attention.
- [x] Support simple mask.
- [x] Support Q/K/V inputs.
- [x] Support f32 accumulation.
- [x] Use softmax.
- [x] Support value aggregation.
- [x] Support non-paged KV metadata where implemented.
- [x] Reject paged KV if unsupported.
- [x] Reject flash attention.
- [x] Reject sliding window if unsupported.
- [x] Reject block sparse attention.
- [x] Reject GQA/MQA if not implemented.
- [x] Add attention scope tests.

## 12. RoPE Scope

- [x] Support baseline RoPE mode.
- [x] Validate base.
- [x] Validate scale.
- [x] Validate dimension.
- [x] Validate position indices.
- [x] Validate tensor shape.
- [x] Validate dtype.
- [x] Reject unsupported variants.
- [x] Add RoPE scope tests.

## 13. MLP Scope

- [x] Express gated MLP using matmul.
- [x] Express gated MLP using silu.
- [x] Express gated MLP using mul.
- [x] Express output projection using matmul.
- [x] Keep fused MLP optional.
- [x] Keep GELU placeholder unless needed.
- [x] Add MLP graph tests.

## 14. Elementwise Scope

- [x] Support add.
- [x] Support mul.
- [x] Support residual-add.
- [x] Define broadcasting rules.
- [x] Reject unsupported broadcasting.
- [x] Add elementwise scope tests.

## 15. Logits Projection

- [x] Use matmul for logits projection.
- [x] Preserve Sampling Contract.
- [x] Avoid Provider-assisted sampling requirement.
- [x] Add logits projection tests.

## 16. Graph Planning Integration

- [x] Plan required-now operators.
- [x] Reject placeholder operators unless policy allows pending path.
- [x] Reject explicitly unsupported operators.
- [x] Insert explicit dtype conversion where allowed.
- [x] Insert explicit layout conversion where allowed.
- [x] Avoid hidden substitutions.
- [x] Add graph planning tests.

## 17. Model Component Integration

- [x] Require first Model Component baseline to use only scoped operators.
- [x] Reject Model Component requiring unsupported operators.
- [x] Report missing operator support.
- [x] Add model component scope tests.

## 18. Kernel Registry Integration

- [x] Query candidates for required-now operators.
- [x] Ensure missing required kernel reports structured error.
- [x] Ensure placeholders do not create false candidates.
- [x] Add registry scope tests.

## 19. Conformance Fixtures

- [x] Add embedding fixture.
- [x] Add matmul fixture.
- [x] Add RMSNorm fixture.
- [x] Add RoPE fixture.
- [x] Add attention fixture.
- [x] Add softmax fixture.
- [x] Add SiLU fixture.
- [x] Add add fixture.
- [x] Add mul fixture.
- [x] Add residual-add fixture.
- [x] Add dtype conversion fixture.
- [x] Add layout conversion fixture.
- [x] Add invalid shape fixtures.
- [x] Add invalid dtype fixtures.
- [x] Add invalid layout fixtures.

## 20. Error Model

- [x] Define operator-out-of-first-scope error.
- [x] Define operator-placeholder-only error.
- [x] Define operator-explicitly-unsupported error.
- [x] Define first-scope-dtype-unsupported error.
- [x] Define first-scope-layout-unsupported error.
- [x] Define first-scope-shape-unsupported error.
- [x] Define first-scope-attribute-unsupported error.
- [x] Define first-scope-kernel-missing error.
- [x] Define first-scope-conformance-missing error.
- [x] Define first-scope-conformance-failed error.
- [x] Define first-scope-graph-planning-failed error.
- [x] Define internal-first-operator-scope error.

## 21. Observability

- [x] Emit first scope operator accepted observation.
- [x] Emit first scope operator rejected observation.
- [x] Emit placeholder operator encountered observation.
- [x] Emit unsupported operator encountered observation.
- [x] Emit required kernel missing observation.
- [x] Emit dtype unsupported in first scope observation.
- [x] Emit layout unsupported in first scope observation.
- [x] Emit shape unsupported in first scope observation.
- [x] Emit first scope conformance passed observation.
- [x] Emit first scope conformance failed observation.
- [x] Avoid raw tensor/prompt/weight/cache/handle logging.

## 22. Tests

- [x] Test required-now classification.
- [x] Test placeholder classification.
- [x] Test explicitly unsupported classification.
- [x] Test future optimized classification.
- [x] Test Reference CPU coverage for required-now.
- [x] Test missing CPU kernel fails.
- [x] Test f32 compute accepted.
- [x] Test f16 compute placeholder behavior.
- [x] Test contiguous layout accepted.
- [x] Test paged layout placeholder behavior.
- [x] Test attention unsupported variant rejected.
- [x] Test GQA/MQA rejected if not implemented.
- [x] Test model component unsupported operator rejection.
- [x] Test graph planning rejects unsupported operator.
- [x] Test conformance fixture discovery.
- [x] Test raw data not logged.

## 23. Documentation

- [x] Document first operator implementation scope.
- [x] Document required-now operators.
- [x] Document placeholder operators.
- [x] Document explicitly unsupported operators.
- [x] Document future optimized operators.
- [x] Document dtype scope.
- [x] Document layout scope.
- [x] Document shape scope.
- [x] Document attention scope.
- [x] Document RoPE scope.
- [x] Document MLP scope.
- [x] Document conformance scope.
- [x] Document non-goals.

## 24. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Operator Scope tests.
- [x] Run Operator tests.
- [x] Run Reference CPU Provider tests.
- [x] Run Kernel Registry tests.
- [x] Run Execution Graph tests.
- [x] Run Model Component tests where impacted.
- [x] Run Conformance tests.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify first scope is minimal.
- [x] Verify unsupported behavior is explicit.
- [x] Verify required-now operators have CPU baseline coverage.