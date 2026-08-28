# Tasks

## 1. Kernel Artifact Domain

- [x] Add Kernel Source Artifact contract.
- [x] Add Compiled Kernel Artifact contract.
- [x] Add Prepared Kernel contract.
- [x] Keep the three lifecycle entities distinct.
- [x] Document lifecycle transitions.
- [x] Add artifact lifecycle tests.

## 2. Source Format Identity

- [x] Define extensible Kernel Source Format identity.
- [x] Avoid closed `TargetLang` enum.
- [x] Support namespace/name/version representation.
- [x] Add source format validation.
- [x] Add unknown future format compatibility tests.

## 3. Kernel Source Artifact

- [x] Add content digest.
- [x] Add source format.
- [x] Add Operator semantic declaration.
- [x] Add fused Operator declaration where applicable.
- [x] Add dtype constraints.
- [x] Add layout constraints.
- [x] Add shape specialization metadata.
- [x] Add target requirements.
- [x] Add compiler requirements.
- [x] Add provenance metadata.
- [x] Add trust/integrity metadata.
- [x] Add validation tests.

## 4. Generated Kernel Provenance

- [x] Define human-authored provenance.
- [x] Define AI-generated provenance.
- [x] Define optimizer-generated provenance.
- [x] Define compiler-generated provenance.
- [x] Define CI-generated provenance.
- [x] Define vendor-provided provenance.
- [x] Ensure provenance does not imply trust.
- [x] Add provenance/trust tests.

## 5. Compiled Kernel Artifact

- [x] Define compiled artifact identity.
- [x] Add content digest.
- [x] Add source artifact digest where known.
- [x] Add compiled format.
- [x] Add compiler identity.
- [x] Add compiler version.
- [x] Add compiler flags fingerprint.
- [x] Add target architecture metadata.
- [x] Add Provider compatibility metadata.
- [x] Add driver/runtime compatibility metadata.
- [x] Add dtype/layout specialization metadata.
- [x] Add shape specialization metadata.
- [x] Add precision metadata.
- [x] Add determinism metadata.
- [x] Add trust/integrity metadata.
- [x] Add validation tests.

## 6. Prepared Kernel

- [x] Define opaque PreparedKernelId.
- [x] Define Prepared Kernel lifecycle.
- [x] Define Provider ownership.
- [x] Define Device binding.
- [x] Define Kernel Artifact binding.
- [x] Define readiness state.
- [x] Define generation/version metadata.
- [x] Define retirement state.
- [x] Define destroy lifecycle.
- [x] Add Prepared Kernel tests.

## 7. Handle Safety

- [x] Ensure PreparedKernelId does not encode native pointer semantics.
- [x] Prevent native function pointers in Runtime public types.
- [x] Prevent native handles in WIT.
- [x] Prevent native handles in Runtime Inference API.
- [x] Prevent native handles in diagnostics by default.
- [x] Add handle exposure tests.

## 8. Provider Ownership

- [x] Define Provider ownership of native prepared state.
- [x] Keep native executable handles Provider-private.
- [x] Keep Provider responsible for native destruction.
- [x] Define Provider preparation failure semantics.
- [x] Add Provider ownership tests.

## 9. Device Boundary

- [x] Keep Device metadata/status only.
- [x] Do not add source compilation to Device trait.
- [x] Do not add artifact loading to Device trait.
- [x] Do not add Prepared Kernel ownership to Device.
- [x] Add boundary tests.

## 10. Scheduler Boundary

- [x] Prevent Scheduler from compiling kernels.
- [x] Prevent Scheduler from loading executable artifacts.
- [x] Allow readiness/admission decisions based on kernel preparation.
- [x] Add Scheduler boundary tests.

## 11. Kernel Registry Integration

- [x] Allow Registry to reference artifact identity.
- [x] Allow Registry to reference PreparedKernelId.
- [x] Keep Registry metadata-only with respect to native handles.
- [x] Validate Prepared Kernel readiness before dispatch.
- [x] Validate Provider/Device binding.
- [x] Add Registry integration tests.

## 12. Execution Graph Boundary

- [x] Keep graph portable.
- [x] Prevent source code in normal graph nodes.
- [x] Prevent executable binary blobs in graph.
- [x] Prevent PreparedKernelId in portable graph representation.
- [x] Prevent native handles in graph.
- [x] Keep Operator semantic requirements explicit.
- [x] Add graph boundary tests.

## 13. Cold Path

- [x] Define compilation as cold-path operation.
- [x] Define preparation as cold-path operation.
- [x] Define validation as cold-path operation.
- [x] Define specialization as cold-path operation.
- [x] Define optional qualification placeholder.
- [x] Add cold-path tests.

## 14. Hot Path

- [x] Define prepared-kernel execution hot path.
- [x] Prevent normal token decode from synchronously compiling kernels.
- [x] Add `kernel-hot-path-compilation-denied` error.
- [x] Add hot-path conformance tests.

## 15. Model Instance Readiness

- [x] Allow required Prepared Kernels to gate readiness.
- [x] Prevent ready state when mandatory preparation failed.
- [x] Define lazy preparation policy placeholder.
- [x] Add readiness tests.

## 16. Lazy Preparation

- [x] Define explicit lazy-preparation policy.
- [x] Define admission state while preparation occurs.
- [x] Prevent silent blocking compile in decode.
- [x] Add lazy preparation tests.

## 17. Artifact Trust

- [x] Add trust state to Kernel Source Artifact.
- [x] Add trust state to Compiled Kernel Artifact.
- [x] Ensure format does not imply trust.
- [x] Ensure AI-generated provenance does not imply trust.
- [x] Ensure local origin does not imply trust.
- [x] Ensure cache presence does not imply trust.
- [x] Add trust regression tests.

## 18. Semantic Compatibility

- [x] Bind artifact to Operator semantics.
- [x] Support fused Operator groups.
- [x] Validate Operator semantic version.
- [x] Reject semantic mismatch.
- [x] Add semantic compatibility tests.

## 19. Shape Specialization

- [x] Add exact-shape specialization.
- [x] Add bounded-shape specialization.
- [x] Add batch range metadata.
- [x] Add sequence range metadata.
- [x] Add attention-specific dimension metadata.
- [x] Add alignment metadata.
- [x] Add shape compatibility tests.

## 20. DType And Layout Specialization

- [x] Add explicit dtype specialization.
- [x] Add explicit layout specialization.
- [x] Prevent hidden dtype conversion.
- [x] Prevent hidden layout conversion.
- [x] Add specialization tests.

## 21. Precision Metadata

- [x] Add accumulation dtype metadata.
- [x] Add approximate math metadata.
- [x] Add tolerance profile.
- [x] Add deterministic tolerance profile.
- [x] Add reduction ordering assumptions.
- [x] Add fused semantics metadata.
- [x] Add precision validation tests.

## 22. Future Cache Compatibility

- [x] Define metadata required for future cache keys.
- [x] Include source digest.
- [x] Include compiler identity/version.
- [x] Include compiler flags fingerprint.
- [x] Include Provider version.
- [x] Include target architecture.
- [x] Include runtime/driver compatibility.
- [x] Include dtype/layout.
- [x] Include shape specialization.
- [x] Do not define eviction policy yet.

## 23. Artifact Replacement

- [x] Allow multiple Kernel Artifact versions.
- [x] Keep Provider loaded during artifact replacement.
- [x] Add artifact version coexistence tests.

## 24. Prepared Kernel Generations

- [x] Allow multiple Prepared Kernel generations.
- [x] Track active references.
- [x] Prevent destruction while in use.
- [x] Allow new invocations to use new generation.
- [x] Add retirement lifecycle tests.

## 25. Provider Lifetime Independence

- [x] Ensure kernel replacement does not unload Provider.
- [x] Ensure Provider state can outlive individual kernels.
- [x] Add Provider lifetime tests.

## 26. Memory Boundary

- [x] Keep executable kernel memory separate from Tensor Resource memory.
- [x] Preserve Memory Manager ownership of inference allocations.
- [x] Prevent kernel preparation from taking ownership of Runtime tensors.
- [x] Add memory boundary tests.

## 27. Runtime Inference API Boundary

- [x] Prevent generation requests from carrying kernel source.
- [x] Prevent generation requests from carrying compiled binaries.
- [x] Prevent generation requests from carrying PreparedKernelId.
- [x] Prevent generation requests from carrying native handles.
- [x] Add inference API boundary tests.

## 28. Component Boundary

- [x] Prevent normal Components from injecting arbitrary kernel source.
- [x] Reserve future authorized kernel import workflows.
- [x] Keep inference Component authority scoped.
- [x] Add Component boundary tests.

## 29. External Generator Boundary

- [x] Document generators as external producers.
- [x] Do not depend on KernelEvolve or any specific generator.
- [x] Support human-generated artifacts.
- [x] Support AI-generated artifacts.
- [x] Support CI-generated artifacts.
- [x] Add generator-neutral tests.

## 30. Error Model

- [x] Add kernel-artifact-invalid.
- [x] Add kernel-artifact-digest-mismatch.
- [x] Add kernel-artifact-format-unsupported.
- [x] Add kernel-artifact-untrusted.
- [x] Add kernel-artifact-operator-incompatible.
- [x] Add kernel-artifact-dtype-incompatible.
- [x] Add kernel-artifact-layout-incompatible.
- [x] Add kernel-artifact-shape-incompatible.
- [x] Add kernel-artifact-target-incompatible.
- [x] Add kernel-artifact-provider-incompatible.
- [x] Add kernel-artifact-driver-incompatible.
- [x] Add kernel-artifact-compiler-incompatible.
- [x] Add kernel-preparation-unavailable.
- [x] Add kernel-preparation-failed.
- [x] Add kernel-prepared-handle-invalid.
- [x] Add kernel-prepared-generation-in-use.
- [x] Add kernel-prepared-destroy-failed.
- [x] Add kernel-prepared-not-ready.
- [x] Add kernel-hot-path-compilation-denied.
- [x] Add internal-kernel-artifact-error.

## 31. Observability

- [x] Observe artifact discovered.
- [x] Observe artifact validated.
- [x] Observe compiled artifact selected.
- [x] Observe preparation started.
- [x] Observe preparation completed.
- [x] Observe preparation failed.
- [x] Observe Prepared Kernel registered.
- [x] Observe Prepared Kernel selected.
- [x] Observe Prepared Kernel retired.
- [x] Observe Prepared Kernel destroyed.
- [x] Observe artifact replacement.
- [x] Observe hot-path compilation denial.
- [x] Redact raw source by default.
- [x] Redact compiled binary bytes.
- [x] Redact native handles.
- [x] Redact local paths/secrets.

## 32. Conformance

- [x] Validate lifecycle entity separation.
- [x] Validate Device does not compile.
- [x] Validate Scheduler does not compile.
- [x] Validate Provider owns prepared native state.
- [x] Validate PreparedKernelId opacity.
- [x] Validate Registry contains no native pointers.
- [x] Validate hot-path compilation denial.
- [x] Validate provenance does not imply trust.
- [x] Validate format does not imply trust.
- [x] Validate Operator semantic compatibility.
- [x] Validate explicit specialization metadata.
- [x] Validate Provider lifetime independence.
- [x] Validate Prepared Kernel generation coexistence.

## 33. Documentation

- [x] Document Kernel Source Artifact.
- [x] Document Compiled Kernel Artifact.
- [x] Document Prepared Kernel.
- [x] Document cold/hot path.
- [x] Document Provider ownership.
- [x] Document Device boundary.
- [x] Document Scheduler boundary.
- [x] Document external generator boundary.
- [x] Document non-goals.

## 34. Final Validation

- [x] Run OpenSpec validation.
- [x] Verify existing Kernel semantics remain intact.
- [x] Verify no native handle escapes Provider.
- [x] Verify Device API remains hardware-focused.
- [x] Verify Scheduler remains orchestration-only.
- [x] Verify no compilation on normal inference hot path.