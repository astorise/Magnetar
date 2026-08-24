# Tasks

## 1. Tensor Scope

- [x] Define Tensor Descriptor.
- [x] Define Tensor Resource.
- [x] Define Tensor View.
- [x] Define Tensor Layout.
- [x] Document Tensor Descriptor versus Tensor Resource.
- [x] Document Tensor Resource versus Memory Allocation.
- [x] Document Tensor Resource versus Provider-owned storage.
- [x] Document Tensor Resource versus KV Cache.

## 2. Tensor Module

- [x] Create first-class `tensor` module or equivalent.
- [x] Export canonical tensor types from crate root.
- [x] Keep tensor contract platform-neutral.
- [x] Keep tensor contract independent from direct Provider selection.
- [x] Add module-level documentation.

## 3. Tensor Descriptor

- [x] Define TensorDescriptor.
- [x] Include shape.
- [x] Include rank.
- [x] Include dtype.
- [x] Include storage dtype.
- [x] Include compute dtype.
- [x] Include layout descriptor.
- [x] Include memory class intent.
- [x] Include mutability intent.
- [x] Include aliasing intent.
- [x] Include Resource Affinity constraints.
- [x] Include semantic role where relevant.
- [x] Prevent raw pointer fields.
- [x] Add descriptor tests.

## 4. Tensor Resource

- [x] Define TensorResourceId.
- [x] Define descriptor binding.
- [x] Define allocation reference.
- [x] Define provider-owned resource reference placeholder.
- [x] Define residency metadata.
- [x] Define memory class.
- [x] Define Resource Affinity.
- [x] Define lifecycle state.
- [x] Define readiness state.
- [x] Define aliasing metadata.
- [x] Define view metadata.
- [x] Define owner subsystem metadata.
- [x] Define observability correlation.
- [x] Add resource tests.

## 5. Tensor Resource Identity

- [x] Ensure TensorResourceId is Runtime-issued.
- [x] Ensure TensorResourceId is opaque.
- [x] Ensure it does not encode raw pointers.
- [x] Ensure it does not encode Provider handles.
- [x] Ensure it does not encode Device handles.
- [x] Ensure it does not encode allocation addresses.
- [x] Ensure it does not encode file paths.
- [x] Ensure it does not encode prompt data.
- [x] Ensure ID alone does not grant unauthorized access.
- [x] Add identity tests.

## 6. Lifecycle

- [x] Define declared state.
- [x] Define planned state.
- [x] Define allocating state.
- [x] Define ready state.
- [x] Define in-use state.
- [x] Define view state.
- [x] Define mutating state.
- [x] Define released state.
- [x] Define evicted state.
- [x] Define invalid state.
- [x] Define failed state.
- [x] Define allowed transitions.
- [x] Add lifecycle tests.

## 7. Readiness

- [x] Define not-ready readiness.
- [x] Define ready readiness.
- [x] Define pending-transfer readiness.
- [x] Define pending-conversion readiness.
- [x] Define pending-compute readiness.
- [x] Define invalid readiness.
- [x] Define failed readiness.
- [x] Keep readiness distinct from lifecycle.
- [x] Add readiness tests.

## 8. Shape Contract

- [x] Define tensor rank.
- [x] Define dimensions.
- [x] Define symbolic dimensions.
- [x] Define dynamic dimension markers.
- [x] Define maximum dimension constraints.
- [x] Define batch dimension role.
- [x] Define sequence dimension role.
- [x] Define hidden dimension role.
- [x] Define head dimension role.
- [x] Validate shape before dispatch where possible.
- [x] Add shape tests.

## 9. DType Contract

- [x] Define storage dtype.
- [x] Define compute dtype.
- [x] Define accumulation dtype.
- [x] Define output dtype.
- [x] Define index dtype.
- [x] Define mask dtype.
- [x] Prevent silent dtype conversion.
- [x] Add dtype tests.

## 10. Layout Contract

- [x] Define TensorLayout.
- [x] Define LayoutDescriptor.
- [x] Define contiguous layout.
- [x] Define strided layout.
- [x] Define blocked layout placeholder.
- [x] Define paged layout placeholder.
- [x] Define packed quantized layout placeholder.
- [x] Define attention-specific layout.
- [x] Define provider-owned opaque layout.
- [x] Define browser-compatible layout.
- [x] Add layout tests.

## 11. Contiguous Layout

- [x] Define dense contiguous storage.
- [x] Define row-major order or explicit dimension order.
- [x] Validate contiguous shape/stride relationship.
- [x] Support contiguous layout in Reference CPU Provider.
- [x] Add contiguous layout tests.

## 12. Strided Layout

- [x] Define explicit strides.
- [x] Define offset.
- [x] Define stride validation.
- [x] Mark unsupported in first scope unless implemented.
- [x] Add strided placeholder tests.

## 13. Blocked Layout

- [x] Reserve blocked layout metadata.
- [x] Define tile/block dimensions.
- [x] Define future support marker.
- [x] Prevent portable Component assumptions.
- [x] Add blocked placeholder tests.

## 14. Paged Layout

- [x] Define page size.
- [x] Define block size.
- [x] Define logical-to-physical metadata.
- [x] Define capacity.
- [x] Define current length.
- [x] Define append behavior.
- [x] Prevent raw page pointers.
- [x] Mark first scope placeholder.
- [x] Add paged layout tests.

## 15. Packed Quantized Layout

- [x] Define quantization method.
- [x] Define bits per value.
- [x] Define group size.
- [x] Define scale dtype.
- [x] Define zero point dtype.
- [x] Define packing order.
- [x] Define block/group metadata.
- [x] Define dequantization requirements.
- [x] Mark first scope placeholder.
- [x] Add packed quantized layout tests.

## 16. Provider-Owned Opaque Layout

- [x] Define opaque layout metadata.
- [x] Prevent raw Provider handle exposure.
- [x] Prevent opaque layout internals in portable Component APIs.
- [x] Require Provider-owned execution path.
- [x] Require Resource Affinity validation.
- [x] Add opaque layout tests.

## 17. Tensor View

- [x] Define TensorView.
- [x] Reference base Tensor Resource ID.
- [x] Define view shape.
- [x] Define view offset.
- [x] Define view strides.
- [x] Define view layout.
- [x] Define dtype compatibility.
- [x] Define mutability.
- [x] Define aliasing relationship.
- [x] Define lifetime dependency.
- [x] Define Resource Affinity inheritance.
- [x] Prevent view outliving base resource.
- [x] Add view tests.

## 18. Aliasing

- [x] Define no-alias.
- [x] Define read-only-alias.
- [x] Define mutable-alias.
- [x] Define input-output-alias.
- [x] Define view-alias.
- [x] Define internal-temporary-alias.
- [x] Validate aliasing before dispatch.
- [x] Add aliasing tests.

## 19. Mutability

- [x] Define immutable.
- [x] Define mutable.
- [x] Define single-writer.
- [x] Define multi-reader.
- [x] Define runtime-internal.
- [x] Define provider-owned.
- [x] Validate mutation before scheduling and dispatch.
- [x] Add mutability tests.

## 20. Memory Class

- [x] Define host memory class.
- [x] Define pinned-host memory class.
- [x] Define device memory class.
- [x] Define unified memory class.
- [x] Define shared memory class.
- [x] Define provider-owned memory class.
- [x] Define browser-linear-memory class.
- [x] Define future-webgpu-buffer memory class.
- [x] Add memory class tests.

## 21. Residency

- [x] Track memory class.
- [x] Track Provider affinity.
- [x] Track Device affinity.
- [x] Track host visibility.
- [x] Track transfer state.
- [x] Track conversion state.
- [x] Track eviction eligibility.
- [x] Track size estimate.
- [x] Track ownership metadata.
- [x] Add residency tests.

## 22. Resource Affinity

- [x] Derive Resource Affinity from residency.
- [x] Prevent caller-forged Provider affinity.
- [x] Prevent caller-forged Device affinity.
- [x] Preserve affinity during Kernel selection.
- [x] Require explicit movement or conversion for affinity changes.
- [x] Add Resource Affinity tests.

## 23. Size Accounting

- [x] Compute size from shape and dtype.
- [x] Compute size from layout metadata.
- [x] Compute size from packing metadata.
- [x] Handle unknown size conservatively.
- [x] Integrate with Memory Manager admission.
- [x] Add size accounting tests.

## 24. Tensor Conversion

- [x] Define dtype conversion.
- [x] Define layout conversion.
- [x] Define memory class movement.
- [x] Define device transfer.
- [x] Define host staging.
- [x] Define opaque layout materialization.
- [x] Define dequantization.
- [x] Define quantization.
- [x] Ensure conversions are explicit.
- [x] Add conversion tests.

## 25. Tensor Materialization

- [x] Materialize from Model Artifact weights.
- [x] Materialize from Adapter Artifact tensors.
- [x] Materialize from input tokens.
- [x] Materialize from KV cache output.
- [x] Materialize from operator output.
- [x] Materialize from Provider-owned output.
- [x] Materialize from test fixture data.
- [x] Track materialization via Memory Manager.
- [x] Add materialization tests.

## 26. Access Boundary

- [x] Prevent default raw tensor storage access by Components.
- [x] Allow descriptor-level access.
- [x] Define future portable safe representation Capability placeholder.
- [x] Allow Provider native storage access only through Runtime-created
      invocation.
- [x] Add access boundary tests.

## 27. Runtime Tensor APIs

- [x] Expose stable metadata.
- [x] Expose controlled resource references.
- [x] Prevent raw pointers.
- [x] Prevent native handles.
- [x] Prevent allocation addresses.
- [x] Prevent Provider internals.
- [x] Prevent Device internals.
- [x] Prevent raw KV cache contents.
- [x] Prevent raw model weights.
- [x] Prevent raw prompts by default.
- [x] Add API tests.

## 28. Execution Graph Integration

- [x] Reference Tensor Descriptors in graph edges.
- [x] Reference Tensor Resources in planned graph.
- [x] Validate shape.
- [x] Validate dtype.
- [x] Validate layout.
- [x] Validate memory behavior.
- [x] Validate aliasing.
- [x] Validate Resource Affinity.
- [x] Validate lifecycle constraints.
- [x] Add graph integration tests.

## 29. Operator Integration

- [x] Operators consume Tensor Descriptors or Resources.
- [x] Operators produce Tensor Descriptors or Resources.
- [x] Validate shape.
- [x] Validate dtype.
- [x] Validate layout.
- [x] Validate aliasing.
- [x] Validate memory behavior.
- [x] Add operator integration tests.

## 30. Kernel Integration

- [x] Kernels receive Runtime-created resource references.
- [x] Validate tensor metadata before dispatch.
- [x] Update readiness after dispatch.
- [x] Update residency after dispatch.
- [x] Update aliasing metadata where relevant.
- [x] Prevent raw public pointer exposure.
- [x] Add kernel integration tests.

## 31. Provider Integration

- [x] Allow Provider-owned opaque tensor storage.
- [x] Keep Provider-owned storage opaque.
- [x] Track metadata needed for future operations.
- [x] Account Provider-owned resources where possible.
- [x] Add provider integration tests.

## 32. Reference CPU Integration

- [x] Support host contiguous Tensor Resources.
- [x] Reject unsupported layouts.
- [x] Reject unsupported dtypes.
- [x] Reject unsupported memory classes.
- [x] Require explicit conversion where policy allows.
- [x] Add Reference CPU tensor tests.

## 33. Browser Compatibility

- [x] Define browser-linear-memory behavior.
- [x] Define JavaScript-mediated buffer placeholder.
- [x] Define future WebGPU buffer placeholder.
- [x] Avoid Wasmtime dependency.
- [x] Avoid native Provider loading requirement.
- [x] Return browser-feature-unsupported where needed.
- [x] Add wasm32 check where feasible.

## 34. Error Model

- [x] Define tensor-descriptor-invalid error.
- [x] Define tensor-resource-not-found error.
- [x] Define tensor-resource-not-ready error.
- [x] Define tensor-resource-invalid error.
- [x] Define tensor-resource-released error.
- [x] Define tensor-shape-invalid error.
- [x] Define tensor-shape-mismatch error.
- [x] Define tensor-rank-unsupported error.
- [x] Define tensor-dtype-unsupported error.
- [x] Define tensor-dtype-conversion-required error.
- [x] Define tensor-dtype-conversion-unsupported error.
- [x] Define tensor-layout-unsupported error.
- [x] Define tensor-layout-conversion-required error.
- [x] Define tensor-layout-conversion-unsupported error.
- [x] Define tensor-memory-class-unsupported error.
- [x] Define tensor-residency-unavailable error.
- [x] Define tensor-Resource-Affinity-conflict error.
- [x] Define tensor-aliasing-violation error.
- [x] Define tensor-mutability-violation error.
- [x] Define tensor-view-invalid error.
- [x] Define tensor-view-base-unavailable error.
- [x] Define tensor-size-unknown error.
- [x] Define tensor-materialization-failed error.
- [x] Define tensor-transfer-failed error.
- [x] Define tensor-browser-feature-unsupported error.
- [x] Define internal-tensor error.

## 35. Observability

- [x] Emit tensor descriptor created observation.
- [x] Emit tensor resource planned observation.
- [x] Emit tensor resource allocated observation.
- [x] Emit tensor resource ready observation.
- [x] Emit tensor resource view created observation.
- [x] Emit tensor resource used observation.
- [x] Emit tensor resource mutated observation.
- [x] Emit tensor conversion planned observation.
- [x] Emit tensor conversion completed observation.
- [x] Emit tensor conversion failed observation.
- [x] Emit tensor transfer planned observation.
- [x] Emit tensor transfer completed observation.
- [x] Emit tensor transfer failed observation.
- [x] Emit tensor released observation.
- [x] Emit tensor evicted observation.
- [x] Emit tensor invalidated observation.
- [x] Emit tensor aliasing violation observation.
- [x] Emit tensor Resource Affinity conflict observation.
- [x] Avoid raw tensor/prompt/weight/cache/handle logging.

## 36. Conformance

- [x] Add Tensor Descriptor conformance tests.
- [x] Add Tensor Resource lifecycle conformance tests.
- [x] Add layout conformance tests.
- [x] Add dtype conformance tests.
- [x] Add aliasing conformance tests.
- [x] Add view lifetime conformance tests.
- [x] Add Resource Affinity conformance tests.
- [x] Add Reference CPU contiguous tensor conformance tests.

## 37. Documentation

- [x] Document Tensor Resource and Layout Contract.
- [x] Document descriptor versus resource.
- [x] Document Tensor Resource lifecycle.
- [x] Document readiness.
- [x] Document shape contract.
- [x] Document dtype contract.
- [x] Document layout contract.
- [x] Document views.
- [x] Document aliasing.
- [x] Document mutability.
- [x] Document memory classes.
- [x] Document residency.
- [x] Document Resource Affinity.
- [x] Document conversions.
- [x] Document Runtime Tensor APIs.
- [x] Document Provider-owned opaque storage.
- [x] Document Reference CPU layout support.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 38. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Tensor tests.
- [x] Run Memory Manager tests.
- [x] Run Operator tests where impacted.
- [x] Run Kernel tests where impacted.
- [x] Run Execution Graph tests where impacted.
- [x] Run Reference CPU Provider tests.
- [x] Run Conformance tests.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify no raw pointers are exposed.
- [x] Verify layout conversion is explicit.
- [x] Verify dtype conversion is explicit.
- [x] Verify Resource Affinity is authoritative.