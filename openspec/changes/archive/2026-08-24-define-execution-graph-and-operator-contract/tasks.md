# Tasks

## 1. Execution Graph Scope

- [x] Define Execution Graph as Runtime-understandable inference graph.
- [x] Document Execution Graph versus Model Component.
- [x] Document Execution Graph versus Provider.
- [x] Document Execution Graph versus Kernel.
- [x] Document Execution Graph versus Scheduler.
- [x] Document Execution Graph versus Memory Manager.

## 2. Operator Scope

- [x] Define Operator as portable semantic operation.
- [x] Document Operator versus Kernel.
- [x] Document Operator versus Provider Capability.
- [x] Document Operator versus model architecture.
- [x] Document Operator versus Sampling.
- [x] Document Operator versus KV Cache.

## 3. Modules

- [x] Create first-class `execution_graph` module or equivalent.
- [x] Create first-class `operator` module or equivalent.
- [x] Export canonical graph and operator types from crate root.
- [x] Keep contracts platform-neutral.
- [x] Keep contracts independent from direct Provider selection.
- [x] Add module-level documentation.

## 4. Operator Catalog

- [x] Define operator catalog.
- [x] Define catalog version.
- [x] Define tensor operator family.
- [x] Define linear algebra operator family.
- [x] Define normalization operator family.
- [x] Define position encoding operator family.
- [x] Define attention operator family.
- [x] Define activation operator family.
- [x] Define quantization operator family.
- [x] Define layout operator family.
- [x] Define sampling-support operator family.
- [x] Define control operator family.
- [x] Add catalog tests.

## 5. Operator Identity

- [x] Define OperatorId.
- [x] Include namespace.
- [x] Include operator name.
- [x] Include semantic version.
- [x] Include operator family.
- [x] Include input contract.
- [x] Include output contract.
- [x] Include attribute schema.
- [x] Include shape rules.
- [x] Include dtype rules.
- [x] Include layout rules.
- [x] Include memory behavior.
- [x] Include determinism metadata.
- [x] Add identity tests.

## 6. Initial Operators

- [x] Define matmul operator.
- [x] Define batched-matmul operator.
- [x] Define embedding operator.
- [x] Define rmsnorm operator.
- [x] Define layernorm operator.
- [x] Define rope operator.
- [x] Define attention operator.
- [x] Define paged-attention placeholder.
- [x] Define softmax operator.
- [x] Define activation operator.
- [x] Define gelu operator.
- [x] Define silu operator.
- [x] Define add operator.
- [x] Define mul operator.
- [x] Define residual-add operator.
- [x] Define dtype-conversion operator.
- [x] Define layout-conversion operator.
- [x] Define quantize operator.
- [x] Define dequantize operator.
- [x] Define sampling-helper operator placeholder.

## 7. Operator Attributes

- [x] Define operator attribute schema.
- [x] Validate matmul attributes.
- [x] Validate attention attributes.
- [x] Validate rope attributes.
- [x] Validate normalization attributes.
- [x] Validate activation attributes.
- [x] Validate quantization attributes.
- [x] Prevent Provider selection attributes.
- [x] Prevent Device selection attributes.
- [x] Add attribute tests.

## 8. Tensor Edges

- [x] Define TensorEdgeId.
- [x] Define logical tensor ID.
- [x] Define shape.
- [x] Define dtype.
- [x] Define layout.
- [x] Define memory class.
- [x] Define residency constraints.
- [x] Define Resource Affinity.
- [x] Define mutability.
- [x] Define lifetime hint.
- [x] Define aliasing behavior.
- [x] Define producer operator.
- [x] Define consumer operators.
- [x] Prevent raw pointer exposure.
- [x] Add tensor edge tests.

## 9. Shape Contract

- [x] Define static shape metadata.
- [x] Define dynamic shape metadata.
- [x] Define shape inference placeholder.
- [x] Define operator shape validation.
- [x] Define shape mismatch error.
- [x] Define shape unsupported error.
- [x] Add shape tests.

## 10. DType Contract

- [x] Define input dtype.
- [x] Define output dtype.
- [x] Define storage dtype.
- [x] Define compute dtype.
- [x] Define accumulation dtype.
- [x] Validate dtype combinations.
- [x] Reject unsupported dtype combinations.
- [x] Add dtype tests.

## 11. Layout Contract

- [x] Define contiguous layout.
- [x] Define strided layout.
- [x] Define blocked layout.
- [x] Define paged layout.
- [x] Define provider-specific opaque layout metadata.
- [x] Define quantized packed layout.
- [x] Define attention-specific layout.
- [x] Define browser-compatible layout.
- [x] Ensure opaque layouts do not leak to Components.
- [x] Define explicit layout conversion.
- [x] Add layout tests.

## 12. Memory Behavior

- [x] Define reads-input behavior.
- [x] Define writes-output behavior.
- [x] Define mutates-input behavior.
- [x] Define aliases-output behavior.
- [x] Define workspace requirement.
- [x] Define in-place support.
- [x] Define host-visible requirement.
- [x] Define device-resident requirement.
- [x] Define pinned-memory requirement.
- [x] Define streaming-output support.
- [x] Define paged-KV-cache support.
- [x] Add memory behavior tests.

## 13. Resource Affinity

- [x] Preserve tensor Resource Affinity.
- [x] Validate operator Resource Affinity compatibility.
- [x] Insert explicit movement only through Runtime policy.
- [x] Reject silent data movement.
- [x] Add Resource Affinity tests.

## 14. Execution Graph Identity

- [x] Define ExecutionGraphId.
- [x] Define graph version.
- [x] Define graph phase.
- [x] Define graph producer metadata.
- [x] Define model instance compatibility metadata.
- [x] Define adapter compatibility metadata.
- [x] Define tokenizer dependency metadata where relevant.
- [x] Define graph digest/fingerprint where useful.
- [x] Add graph identity tests.

## 15. Graph Producer Boundary

- [x] Allow Runtime-native graph producer.
- [x] Allow Model Component graph producer.
- [x] Allow Provider-assisted graph builder placeholder.
- [x] Allow test fixture graph producer.
- [x] Validate graph regardless of producer.
- [x] Prevent Component direct Provider access.
- [x] Add producer boundary tests.

## 16. Graph Phases

- [x] Define model-load graph phase.
- [x] Define warmup graph phase.
- [x] Define prefill graph phase.
- [x] Define decode graph phase.
- [x] Define adapter-activation graph phase.
- [x] Define adapter-merge graph phase.
- [x] Define sampling-helper graph phase.
- [x] Define test graph phase.
- [x] Add phase tests.

## 17. Graph Validation

- [x] Validate graph identity.
- [x] Validate graph version.
- [x] Validate operator identities.
- [x] Validate operator attributes.
- [x] Validate input/output arity.
- [x] Validate tensor edge consistency.
- [x] Validate shape compatibility.
- [x] Validate dtype compatibility.
- [x] Validate layout compatibility.
- [x] Validate Resource Affinity.
- [x] Validate memory behavior.
- [x] Validate aliasing rules.
- [x] Validate lifecycle/resource constraints.
- [x] Validate Provider Capability feasibility.
- [x] Validate policy constraints.
- [x] Add graph validation tests.

## 18. Graph Planning

- [x] Determine operator execution order.
- [x] Prepare fusion opportunity placeholders.
- [x] Plan memory allocation needs.
- [x] Plan workspace needs.
- [x] Plan data movement requirements.
- [x] Plan layout conversion requirements.
- [x] Plan dtype conversion requirements.
- [x] Plan KV cache use.
- [x] Plan adapter paths.
- [x] Plan Provider/Device compatibility.
- [x] Prepare kernel selection placeholder.
- [x] Plan batching compatibility.
- [x] Plan failure handling.
- [x] Add graph planning tests.

## 19. Graph Execution Boundary

- [x] Execute graphs through Runtime-owned path.
- [x] Prevent Component direct Provider calls.
- [x] Prevent raw Provider handle exposure.
- [x] Prevent raw Device handle exposure.
- [x] Prevent raw memory pointer exposure.
- [x] Prevent raw tensor storage exposure.
- [x] Add execution boundary tests.

## 20. Prefill And Decode Graphs

- [x] Define prefill graph requirements.
- [x] Define decode graph requirements.
- [x] Allow different scheduling policy.
- [x] Allow different memory behavior.
- [x] Allow KV cache input/output metadata.
- [x] Add prefill/decode graph tests.

## 21. Attention Operator

- [x] Define attention operator metadata.
- [x] Define causal mode.
- [x] Define attention mask kind.
- [x] Define query head count.
- [x] Define key/value head count.
- [x] Define head dimension.
- [x] Define sequence length.
- [x] Define context length.
- [x] Define KV cache usage.
- [x] Define paged cache support.
- [x] Define position encoding dependency.
- [x] Define dtype requirements.
- [x] Define layout requirements.
- [x] Add attention operator tests.

## 22. RoPE Operator

- [x] Define RoPE operator or attribute.
- [x] Define base.
- [x] Define scale.
- [x] Define dimension.
- [x] Define position index mode.
- [x] Define dynamic scaling metadata.
- [x] Define model compatibility.
- [x] Add RoPE tests.

## 23. Normalization Operators

- [x] Define RMSNorm operator.
- [x] Define LayerNorm operator.
- [x] Define epsilon attribute.
- [x] Define normalized dimension.
- [x] Define dtype behavior.
- [x] Define accumulation dtype behavior.
- [x] Add normalization tests.

## 24. Quantization Operators

- [x] Define quantize operator.
- [x] Define dequantize operator.
- [x] Define requantize placeholder.
- [x] Define quantized-matmul placeholder.
- [x] Define unpack placeholder.
- [x] Define pack placeholder.
- [x] Define scale-apply placeholder.
- [x] Validate quantization metadata.
- [x] Add quantization operator tests.

## 25. Adapter-Aware Graphs

- [x] Represent adapter overlay path.
- [x] Represent adapter merge graph.
- [x] Represent provider-fused adapter path placeholder.
- [x] Include adapter set in graph identity where semantics change.
- [x] Invalidate dependent caches on adapter semantic change.
- [x] Add adapter graph tests.

## 26. KV-Cache-Aware Graphs

- [x] Represent KV cache inputs.
- [x] Represent KV cache outputs.
- [x] Represent KV cache append behavior.
- [x] Represent paged cache metadata.
- [x] Validate KV cache compatibility.
- [x] Prevent raw KV cache exposure.
- [x] Add KV cache graph tests.

## 27. Prefix-Cache-Aware Graphs

- [x] Represent reused prefix length.
- [x] Represent backing KV cache reference.
- [x] Adjust prefill boundary.
- [x] Validate prefix reuse before graph execution.
- [x] Add prefix graph tests.

## 28. Sampling Helper Graphs

- [x] Represent sampling helper operations where useful.
- [x] Preserve Sampling Contract ownership of token selection.
- [x] Prevent graph operators from owning full sampling semantics.
- [x] Add sampling helper tests.

## 29. Determinism Metadata

- [x] Define operator determinism metadata.
- [x] Include dtype influence.
- [x] Include Provider influence.
- [x] Include Device influence.
- [x] Include kernel implementation influence.
- [x] Include reduction behavior.
- [x] Include memory layout influence.
- [x] Surface determinism to Generation/Sampling.
- [x] Add determinism tests.

## 30. Error Model

- [x] Define operator-not-found error.
- [x] Define operator-version-unsupported error.
- [x] Define operator-attribute-invalid error.
- [x] Define input-arity-invalid error.
- [x] Define output-arity-invalid error.
- [x] Define shape-mismatch error.
- [x] Define shape-unsupported error.
- [x] Define dtype-unsupported error.
- [x] Define dtype-conversion-required error.
- [x] Define dtype-conversion-unsupported error.
- [x] Define layout-unsupported error.
- [x] Define layout-conversion-required error.
- [x] Define layout-conversion-unsupported error.
- [x] Define memory-behavior-unsupported error.
- [x] Define workspace-unavailable error.
- [x] Define Resource-Affinity-conflict error.
- [x] Define Provider-capability-unavailable error.
- [x] Define kernel-unavailable error placeholder.
- [x] Define graph-validation-failed error.
- [x] Define graph-planning-failed error.
- [x] Define graph-execution-failed error.
- [x] Define browser-feature-unsupported error.
- [x] Define internal-operator error.

## 31. Observability

- [x] Emit graph created observation.
- [x] Emit graph validation started observation.
- [x] Emit graph validation failed observation.
- [x] Emit graph validated observation.
- [x] Emit graph planning started observation.
- [x] Emit graph planning completed observation.
- [x] Emit operator planned observation.
- [x] Emit data movement inserted observation.
- [x] Emit dtype conversion inserted observation.
- [x] Emit layout conversion inserted observation.
- [x] Emit workspace requested observation.
- [x] Emit graph execution started observation.
- [x] Emit operator execution started observation.
- [x] Emit operator execution completed observation.
- [x] Emit operator execution failed observation.
- [x] Emit graph execution completed observation.
- [x] Emit graph execution failed observation.
- [x] Avoid raw tensor/prompt/weight/cache/handle logging.

## 32. Browser Compatibility

- [x] Keep graph/operator contracts platform-neutral.
- [x] Avoid Wasmtime dependency.
- [x] Avoid native Provider loading requirement.
- [x] Account for WebAssembly linear memory.
- [x] Account for future WebGPU buffers.
- [x] Return unsupported browser errors where needed.
- [x] Add wasm32 check where feasible.

## 33. Tests

- [x] Test operator identity.
- [x] Test operator attribute validation.
- [x] Test shape validation.
- [x] Test dtype validation.
- [x] Test layout validation.
- [x] Test memory behavior validation.
- [x] Test Resource Affinity conflict.
- [x] Test graph validation success.
- [x] Test graph validation failure.
- [x] Test graph planning inserts dtype conversion.
- [x] Test graph planning inserts layout conversion.
- [x] Test graph planning rejects silent movement.
- [x] Test Component-produced graph is validated.
- [x] Test prefill graph.
- [x] Test decode graph.
- [x] Test attention operator metadata.
- [x] Test RoPE metadata.
- [x] Test adapter-aware graph identity.
- [x] Test KV-cache-aware graph validation.
- [x] Test prefix-aware prefill boundary.
- [x] Test raw handles not exposed.
- [x] Test raw tensor values not logged.

## 34. Documentation

- [x] Document Execution Graph.
- [x] Document Operator Contract.
- [x] Document Operator versus Kernel.
- [x] Document graph producer boundary.
- [x] Document operator catalog.
- [x] Document tensor edges.
- [x] Document shape contract.
- [x] Document dtype contract.
- [x] Document layout contract.
- [x] Document memory behavior.
- [x] Document Resource Affinity.
- [x] Document graph validation.
- [x] Document graph planning.
- [x] Document prefill/decode graphs.
- [x] Document attention operator.
- [x] Document adapter-aware graphs.
- [x] Document KV-cache-aware graphs.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 35. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Execution Graph tests.
- [x] Run Operator tests.
- [x] Run Model Instance tests where impacted.
- [x] Run Adapter tests where impacted.
- [x] Run KV Cache tests where impacted.
- [x] Run Prefix Cache tests where impacted.
- [x] Run Memory Manager tests.
- [x] Run Provider conformance tests where impacted.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify Operators are not kernels.
- [x] Verify graphs do not call Providers directly from Components.
- [x] Verify graph execution preserves Resource Affinity.
- [x] Verify no raw handles or tensor storage are exposed.