# Tasks

## 1. Qwen Baseline Scope

- [x] Define Qwen Model Component baseline.
- [x] Document Qwen Component versus Qwen Provider.
- [x] Document Qwen Component versus Kernel.
- [x] Document Qwen Component versus Model Artifact.
- [x] Document first baseline limitations.
- [x] Document non-goals.

## 2. Qwen Component Module

- [x] Create `qwen_model_component` module or equivalent.
- [x] Support Runtime-native implementation path.
- [x] Support WASM Component path placeholder.
- [x] Support test fixture implementation path.
- [x] Keep component platform-neutral.
- [x] Add module-level documentation.

## 3. Identity And Versioning

- [x] Define QwenModelComponentId.
- [x] Define Qwen baseline contract version.
- [x] Define supported Model Component contract version.
- [x] Define supported Model Artifact schema version.
- [x] Define supported Operator catalog version.
- [x] Define supported Execution Graph contract version.
- [x] Define supported Tensor contract version.
- [x] Define supported Tokenizer contract version.
- [x] Define supported KV cache contract version.
- [x] Define supported Adapter contract version where relevant.
- [x] Add version tests.

## 4. Architecture Metadata

- [x] Validate architecture family.
- [x] Validate model type decoder-only.
- [x] Validate hidden size.
- [x] Validate layer count.
- [x] Validate attention head count.
- [x] Validate KV head count.
- [x] Validate head dimension.
- [x] Validate hidden size/head configuration.
- [x] Validate intermediate size.
- [x] Validate vocabulary size.
- [x] Validate context length.
- [x] Validate activation kind.
- [x] Validate normalization kind.
- [x] Validate attention variant.
- [x] Add architecture metadata tests.

## 5. Config Validation

- [x] Accept Runtime-provided validated config metadata.
- [x] Reject arbitrary config path reads.
- [x] Validate required config fields.
- [x] Reject unsupported config schema.
- [x] Reject unsupported features.
- [x] Add config validation tests.

## 6. Model Artifact Compatibility

- [x] Validate artifact architecture family.
- [x] Validate artifact schema version.
- [x] Validate weight tensor inventory metadata.
- [x] Validate tensor naming or logical mapping.
- [x] Validate tokenizer association.
- [x] Validate generation config compatibility.
- [x] Validate quantization metadata.
- [x] Validate adapter compatibility metadata.
- [x] Preserve Runtime artifact trust enforcement.
- [x] Add artifact compatibility tests.

## 7. Tensor Inventory

- [x] Define token_embedding logical tensor.
- [x] Define per-layer input_norm logical tensor.
- [x] Define per-layer q_proj logical tensor.
- [x] Define per-layer k_proj logical tensor.
- [x] Define per-layer v_proj logical tensor.
- [x] Define per-layer o_proj logical tensor.
- [x] Define per-layer post_attn_norm logical tensor.
- [x] Define per-layer gate_proj logical tensor.
- [x] Define per-layer up_proj logical tensor.
- [x] Define per-layer down_proj logical tensor.
- [x] Define final_norm logical tensor.
- [x] Define lm_head logical tensor.
- [x] Support tied embedding metadata.
- [x] Reject missing required tensors.
- [x] Add tensor inventory tests.

## 8. Tensor Shape Validation

- [x] Validate embedding shape.
- [x] Validate q_proj shape.
- [x] Validate k_proj shape.
- [x] Validate v_proj shape.
- [x] Validate o_proj shape.
- [x] Validate gate_proj shape.
- [x] Validate up_proj shape.
- [x] Validate down_proj shape.
- [x] Validate norm weight shape.
- [x] Validate lm_head shape.
- [x] Validate tied embedding shape.
- [x] Add tensor shape tests.

## 9. Target Modules

- [x] Expose q_proj target module.
- [x] Expose k_proj target module.
- [x] Expose v_proj target module.
- [x] Expose o_proj target module.
- [x] Expose gate_proj target module.
- [x] Expose up_proj target module.
- [x] Expose down_proj target module.
- [x] Expose lm_head target module.
- [x] Expose embedding target module.
- [x] Include layer selector metadata.
- [x] Include adapter method compatibility.
- [x] Include graph insertion point.
- [x] Add target module tests.

## 10. Tokenizer Compatibility

- [x] Validate vocabulary size compatibility.
- [x] Validate EOS token availability.
- [x] Validate BOS token policy where relevant.
- [x] Validate pad token policy where relevant.
- [x] Validate special token metadata.
- [x] Validate chat template compatibility metadata where relevant.
- [x] Validate added token behavior where relevant.
- [x] Add tokenizer compatibility tests.

## 11. Generation Defaults Compatibility

- [x] Validate maximum context length.
- [x] Validate EOS token references.
- [x] Validate BOS token policy.
- [x] Treat temperature/top-p/top-k defaults as non-authoritative.
- [x] Validate stop token metadata.
- [x] Add generation metadata tests.

## 12. Required Operators

- [x] Require embedding.
- [x] Require rmsnorm.
- [x] Require matmul.
- [x] Require rope.
- [x] Require attention.
- [x] Require softmax.
- [x] Require silu.
- [x] Require add.
- [x] Require mul.
- [x] Require residual-add.
- [x] Require dtype-conversion where needed.
- [x] Require layout-conversion where needed.
- [x] Reject unsupported required operators.
- [x] Add operator requirement tests.

## 13. Prefill Graph

- [x] Define prefill graph production request.
- [x] Define prefill graph result.
- [x] Include input token IDs.
- [x] Include embedding.
- [x] Include repeated decoder layers.
- [x] Include final RMSNorm.
- [x] Include logits matmul.
- [x] Include KV cache write/append metadata where enabled.
- [x] Validate graph through Runtime.
- [x] Add prefill graph tests.

## 14. Decode Graph

- [x] Define decode graph production request.
- [x] Define decode graph result.
- [x] Support one-token or small-token decode.
- [x] Consume prior KV cache where enabled.
- [x] Produce logits.
- [x] Preserve Sampling boundary.
- [x] Validate graph through Runtime.
- [x] Add decode graph tests.

## 15. Decoder Layer Graph

- [x] Add input RMSNorm.
- [x] Add q_proj matmul.
- [x] Add k_proj matmul.
- [x] Add v_proj matmul.
- [x] Add RoPE on q/k.
- [x] Add attention.
- [x] Add o_proj matmul.
- [x] Add residual-add.
- [x] Add post-attention RMSNorm.
- [x] Add gate_proj matmul.
- [x] Add SiLU.
- [x] Add up_proj matmul.
- [x] Add mul.
- [x] Add down_proj matmul.
- [x] Add residual-add.
- [x] Add decoder layer graph tests.

## 16. Attention Metadata

- [x] Declare causal attention.
- [x] Declare attention mask kind.
- [x] Declare attention head count.
- [x] Declare KV head count.
- [x] Declare head dimension.
- [x] Declare sequence length.
- [x] Declare context length.
- [x] Declare RoPE dependency.
- [x] Declare KV cache behavior.
- [x] Declare dtype requirements.
- [x] Declare layout requirements.
- [x] Reject unsupported GQA/MQA path if not implemented.
- [x] Add attention metadata tests.

## 17. RoPE Metadata

- [x] Declare RoPE base.
- [x] Declare RoPE scale.
- [x] Declare RoPE dimension.
- [x] Declare position index mode.
- [x] Declare context length compatibility.
- [x] Declare dynamic scaling support status.
- [x] Reject unsupported RoPE variants.
- [x] Add RoPE metadata tests.

## 18. MLP Graph

- [x] Build gate projection with matmul.
- [x] Build up projection with matmul.
- [x] Apply SiLU to gate projection.
- [x] Multiply activated gate and up projection.
- [x] Build down projection with matmul.
- [x] Avoid requiring fused MLP.
- [x] Add MLP graph tests.

## 19. Normalization

- [x] Use RMSNorm.
- [x] Validate RMSNorm epsilon.
- [x] Validate normalized dimension.
- [x] Reject LayerNorm requirement for baseline.
- [x] Add normalization tests.

## 20. Logits Projection

- [x] Support lm_head matmul.
- [x] Support tied embedding metadata where implemented.
- [x] Produce logits Tensor Descriptor.
- [x] Preserve Sampling ownership.
- [x] Add logits projection tests.

## 21. KV Cache Metadata

- [x] Declare layer count.
- [x] Declare KV head count.
- [x] Declare attention head count.
- [x] Declare head dimension.
- [x] Declare cache dtype.
- [x] Declare sequence dimension.
- [x] Declare batch dimension.
- [x] Declare position behavior.
- [x] Declare append behavior.
- [x] Declare layout preference.
- [x] Declare paged cache support status.
- [x] Add KV cache metadata tests.

## 22. Prefix Cache Metadata

- [x] Expose architecture family metadata.
- [x] Expose component version metadata.
- [x] Expose tokenizer compatibility influence.
- [x] Expose RoPE metadata influence.
- [x] Expose adapter set influence.
- [x] Expose model config fingerprint influence.
- [x] Add Prefix Cache metadata tests.

## 23. Adapter Compatibility

- [x] Expose LoRA-compatible target modules where supported.
- [x] Validate adapter target module names.
- [x] Validate adapter tensor shapes.
- [x] Declare overlay graph support status.
- [x] Declare merge graph support status.
- [x] Reject unsupported adapter activation.
- [x] Add adapter compatibility tests.

## 24. Quantization Compatibility

- [x] Validate quantization metadata.
- [x] Reject unsupported quantized artifacts.
- [x] Avoid hidden dequantization.
- [x] Declare dequantization path only if implemented.
- [x] Add quantization compatibility tests.

## 25. Tensor Layout And DType

- [x] Target contiguous tensor layout.
- [x] Reject unsupported layouts.
- [x] Require explicit layout conversion.
- [x] Target f32 compute.
- [x] Reject unsupported dtypes.
- [x] Require explicit dtype conversion.
- [x] Add tensor layout/dtype tests.

## 26. Model Loading Integration

- [x] Resolve Qwen Component during Model Loading.
- [x] Use Qwen Component for architecture validation.
- [x] Use Qwen Component for tensor inventory validation.
- [x] Use Qwen Component for target module metadata.
- [x] Use Qwen Component for graph metadata preparation.
- [x] Preserve Runtime trust validation.
- [x] Preserve Memory Manager admission.
- [x] Add Model Loading integration tests.

## 27. Model Instance Integration

- [x] Reference Qwen Component identity from Model Instance.
- [x] Include Qwen Component version in compatibility metadata.
- [x] Include Qwen config fingerprint in cache compatibility.
- [x] Preserve Runtime-owned Model Instance lifecycle.
- [x] Add Model Instance integration tests.

## 28. Generation Integration

- [x] Request prefill graph through Runtime.
- [x] Request decode graph through Runtime.
- [x] Preserve Generation lifecycle.
- [x] Preserve stop conditions.
- [x] Preserve Sampling invocation.
- [x] Preserve streaming semantics.
- [x] Preserve cancellation semantics.
- [x] Add Generation integration tests.

## 29. Reference CPU Integration

- [x] Validate required operators have CPU coverage.
- [x] Validate contiguous f32 CPU path.
- [x] Reject missing CPU operator coverage.
- [x] Reject unsupported attention variant.
- [x] Add Reference CPU execution smoke tests.

## 30. Component Runtime Integration

- [x] Define WASM Component path placeholder.
- [x] Enforce inference-scoped authority.
- [x] Deny filesystem.
- [x] Deny network.
- [x] Deny process.
- [x] Deny shell.
- [x] Deny secrets.
- [x] Deny Git.
- [x] Deny workspace.
- [x] Deny Provider handles.
- [x] Deny Device handles.
- [x] Deny Kernel handles.
- [x] Deny raw tensor pointers.
- [x] Add authority tests.

## 31. Browser Compatibility

- [x] Keep Qwen baseline contract platform-neutral.
- [x] Avoid Wasmtime requirement on browser.
- [x] Avoid native Provider loading requirement.
- [x] Return unsupported browser errors where needed.
- [x] Add wasm32 check where feasible.

## 32. Error Model

- [x] Define qwen-component-not-found error.
- [x] Define qwen-component-invalid error.
- [x] Define qwen-component-untrusted error.
- [x] Define qwen-component-unsupported-version error.
- [x] Define qwen-architecture-unsupported error.
- [x] Define qwen-config-invalid error.
- [x] Define qwen-tensor-inventory-missing error.
- [x] Define qwen-tensor-shape-mismatch error.
- [x] Define qwen-tokenizer-incompatible error.
- [x] Define qwen-generation-metadata-invalid error.
- [x] Define qwen-operator-unsupported error.
- [x] Define qwen-graph-production-failed error.
- [x] Define qwen-graph-validation-failed error.
- [x] Define qwen-target-module-unavailable error.
- [x] Define qwen-adapter-unsupported error.
- [x] Define qwen-KV-cache-metadata-invalid error.
- [x] Define qwen-RoPE-unsupported error.
- [x] Define qwen-attention-variant-unsupported error.
- [x] Define qwen-quantization-unsupported error.
- [x] Define qwen-dtype-unsupported error.
- [x] Define qwen-layout-unsupported error.
- [x] Define qwen-Reference-CPU-coverage-missing error.
- [x] Define qwen-capability-unavailable error.
- [x] Define qwen-authority-denied error.
- [x] Define qwen-browser-feature-unsupported error.
- [x] Define internal-qwen-component error.

## 33. Observability

- [x] Emit Qwen Component resolved observation.
- [x] Emit Qwen Component validated observation.
- [x] Emit Qwen Component rejected observation.
- [x] Emit Qwen config validated observation.
- [x] Emit Qwen tensor inventory checked observation.
- [x] Emit Qwen target modules exposed observation.
- [x] Emit Qwen tokenizer compatibility checked observation.
- [x] Emit Qwen KV metadata produced observation.
- [x] Emit Qwen prefill graph produced observation.
- [x] Emit Qwen decode graph produced observation.
- [x] Emit Qwen graph validation failed observation.
- [x] Emit Qwen required operator missing observation.
- [x] Emit Qwen Reference CPU coverage missing observation.
- [x] Emit Qwen authority denied observation.
- [x] Emit Qwen conformance result observation.
- [x] Avoid raw prompt/weight/adapter/cache/handle logging.

## 34. Conformance

- [x] Add valid minimal Qwen config fixture.
- [x] Add invalid architecture family fixture.
- [x] Add invalid hidden/head configuration fixture.
- [x] Add missing tensor inventory fixture.
- [x] Add invalid tensor shape fixture.
- [x] Add target module exposure fixture.
- [x] Add prefill graph production fixture.
- [x] Add decode graph production fixture.
- [x] Add required operator scope fixture.
- [x] Add tokenizer compatibility fixture.
- [x] Add KV cache metadata fixture.
- [x] Add adapter target validation fixture.
- [x] Add unsupported quantization rejection fixture.
- [x] Add authority denial fixture.
- [x] Add no raw handle exposure fixture.
- [x] Add conformance report.

## 35. Documentation

- [x] Document Qwen Model Component baseline.
- [x] Document Qwen Component versus Qwen Provider.
- [x] Document supported architecture metadata.
- [x] Document config validation.
- [x] Document tensor inventory.
- [x] Document target modules.
- [x] Document tokenizer compatibility.
- [x] Document prefill graph.
- [x] Document decode graph.
- [x] Document decoder layer graph.
- [x] Document attention metadata.
- [x] Document RoPE metadata.
- [x] Document MLP graph.
- [x] Document KV cache metadata.
- [x] Document adapter compatibility.
- [x] Document quantization limitations.
- [x] Document Reference CPU path.
- [x] Document browser limitations.
- [x] Document non-goals.

## 36. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Qwen Component tests.
- [x] Run Model Component tests.
- [x] Run Model Artifact tests.
- [x] Run Model Loading tests.
- [x] Run Model Instance tests.
- [x] Run Execution Graph tests.
- [x] Run Operator Scope tests.
- [x] Run Tensor tests.
- [x] Run KV Cache tests.
- [x] Run Tokenizer tests.
- [x] Run Reference CPU Provider tests.
- [x] Run Conformance tests.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify no QwenProvider is introduced.
- [x] Verify Qwen graphs use portable Operators.
- [x] Verify Reference CPU path is explicit.
- [x] Verify unsupported variants fail explicitly.
