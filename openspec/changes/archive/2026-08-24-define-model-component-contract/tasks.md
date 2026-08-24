# Tasks

## 1. Model Component Scope

- [x] Define Model Component as portable architecture implementation.
- [x] Document Model Component versus Provider.
- [x] Document Model Component versus Kernel.
- [x] Document Model Component versus Operator.
- [x] Document Model Component versus Model Artifact.
- [x] Document Model Component versus Generation.
- [x] Document Model Component versus agent/tool runtime.

## 2. Model Component Module

- [x] Create first-class `model_component` module or equivalent.
- [x] Export canonical Model Component types from crate root.
- [x] Keep contract platform-neutral.
- [x] Keep contract independent from direct Provider/Device selection.
- [x] Add module-level documentation.

## 3. Model Component Identity

- [x] Define ModelComponentId.
- [x] Define component version.
- [x] Define supported architecture families.
- [x] Define supported architecture revisions.
- [x] Define supported Model Artifact schema versions.
- [x] Define supported Runtime Capability versions.
- [x] Define supported Operator catalog version.
- [x] Define supported Execution Graph contract version.
- [x] Define trust status.
- [x] Define provenance metadata.
- [x] Define signature state where applicable.
- [x] Add identity tests.

## 4. Model Component Role

- [x] Mark Model Component as Component Artifact role.
- [x] Support WebAssembly Component implementation.
- [x] Support Runtime-native implementation.
- [x] Support test fixture implementation.
- [x] Support browser-compatible implementation placeholder.
- [x] Validate role before use.
- [x] Add role tests.

## 5. Architecture Compatibility

- [x] Validate architecture family.
- [x] Validate model type.
- [x] Validate hidden size.
- [x] Validate layer count.
- [x] Validate attention head count.
- [x] Validate KV head count.
- [x] Validate head dimension.
- [x] Validate intermediate size.
- [x] Validate vocabulary size.
- [x] Validate context length.
- [x] Validate position encoding.
- [x] Validate normalization kind.
- [x] Validate activation kind.
- [x] Validate attention variant.
- [x] Validate quantization metadata.
- [x] Validate tokenizer compatibility.
- [x] Validate adapter target modules.
- [x] Add architecture compatibility tests.

## 6. Config Validation

- [x] Accept validated Model Artifact metadata from Runtime.
- [x] Accept authorized config data.
- [x] Reject invalid config schema.
- [x] Reject unsupported config values.
- [x] Prevent arbitrary filesystem path reads.
- [x] Add config validation tests.

## 7. Target Modules

- [x] Define target module metadata.
- [x] Define q_proj module role.
- [x] Define k_proj module role.
- [x] Define v_proj module role.
- [x] Define o_proj module role.
- [x] Define gate_proj module role.
- [x] Define up_proj module role.
- [x] Define down_proj module role.
- [x] Define lm_head module role.
- [x] Define embedding module role.
- [x] Define norm module role.
- [x] Define attention module role.
- [x] Define mlp module role.
- [x] Add target module tests.

## 8. Graph Production

- [x] Define graph production request.
- [x] Define graph production result.
- [x] Support model-load graph phase.
- [x] Support warmup graph phase.
- [x] Support prefill graph phase.
- [x] Support decode graph phase.
- [x] Support adapter-activation graph phase.
- [x] Support adapter-merge graph phase.
- [x] Support sampling-helper graph phase.
- [x] Support test graph phase.
- [x] Validate produced graphs through Runtime.
- [x] Add graph production tests.

## 9. Operator Requirements

- [x] Declare required Operator IDs.
- [x] Declare required Operator families.
- [x] Declare required Operator versions.
- [x] Declare optional Operator alternatives.
- [x] Declare shape-related Operator requirements.
- [x] Declare dtype-related Operator requirements.
- [x] Declare layout-related Operator requirements.
- [x] Prevent Provider-specific Kernel names as authoritative requirements.
- [x] Add Operator requirement tests.

## 10. Capability Requirements

- [x] Declare model metadata validation Capability.
- [x] Declare graph production Capability.
- [x] Declare operator catalog read Capability.
- [x] Declare tensor descriptor creation Capability.
- [x] Declare KV cache metadata Capability.
- [x] Declare adapter metadata Capability.
- [x] Declare tokenizer metadata Capability.
- [x] Declare generation defaults validation Capability.
- [x] Declare diagnostics Capability.
- [x] Declare observability emit Capability.
- [x] Add Capability requirement tests.

## 11. Authority Model

- [x] Allow model-artifact-read.
- [x] Allow tokenizer-artifact-read.
- [x] Allow prompt-template-read.
- [x] Allow adapter-artifact-read.
- [x] Allow quantization-artifact-read.
- [x] Allow inference-session-state.
- [x] Allow generation-session-state.
- [x] Allow kv-cache-access.
- [x] Allow prefix-cache-access.
- [x] Allow compute-capability.
- [x] Allow generation-capability.
- [x] Allow sampling-capability.
- [x] Allow observability-emit.
- [x] Allow runtime-diagnostics.
- [x] Allow graph-production.
- [x] Allow operator-catalog-read.
- [x] Deny filesystem.
- [x] Deny network.
- [x] Deny env.
- [x] Deny process.
- [x] Deny shell.
- [x] Deny secrets.
- [x] Deny workspace.
- [x] Deny git.
- [x] Deny source-control.
- [x] Deny tool-execution.
- [x] Deny external-service.
- [x] Add authority tests.

## 12. Provider Boundary

- [x] Prevent raw Provider handle exposure.
- [x] Prevent raw Device handle exposure.
- [x] Prevent raw Kernel handle exposure.
- [x] Prevent raw memory pointer exposure.
- [x] Prevent Provider-owned resource exposure.
- [x] Allow redacted compatibility diagnostics where policy allows.
- [x] Add Provider boundary tests.

## 13. Graph Validation Boundary

- [x] Treat Component-produced graphs as untrusted until validated.
- [x] Validate graph schema.
- [x] Validate graph version.
- [x] Validate graph phase.
- [x] Validate Operator identities.
- [x] Validate Operator attributes.
- [x] Validate tensor edges.
- [x] Validate shape rules.
- [x] Validate dtype rules.
- [x] Validate layout rules.
- [x] Validate Resource Affinity.
- [x] Validate memory behavior.
- [x] Validate adapter metadata.
- [x] Validate KV cache metadata.
- [x] Validate policy constraints.
- [x] Add graph validation boundary tests.

## 14. Model Loading Integration

- [ ] Use Model Component for architecture compatibility.
- [ ] Use Model Component for config validation.
- [ ] Use Model Component for target module declaration.
- [ ] Use Model Component for graph metadata preparation.
- [ ] Use Model Component for warmup graph construction.
- [ ] Prevent bypassing artifact trust.
- [ ] Prevent bypassing memory admission.
- [ ] Add Model Loading integration tests.

## 15. Model Instance Integration

- [ ] Reference Model Component from Model Instance metadata.
- [ ] Preserve Model Instance lifecycle ownership in Runtime.
- [ ] Include Model Component version in instance compatibility.
- [ ] Include Model Component version in cache compatibility where relevant.
- [ ] Add Model Instance integration tests.

## 16. Generation Integration

- [ ] Request prefill graph where available.
- [ ] Request decode graph where available.
- [ ] Preserve Generation stop conditions.
- [ ] Preserve Generation streaming semantics.
- [ ] Preserve Sampling boundary.
- [ ] Prevent Model Component from owning request lifecycle.
- [ ] Add Generation integration tests.

## 17. Adapter Integration

- [x] Expose adapter target modules.
- [x] Validate adapter compatibility metadata.
- [x] Produce adapter overlay graph where supported.
- [x] Produce adapter merge graph where supported.
- [ ] Expose provider-fused adapter metadata placeholder.
- [x] Preserve Runtime-owned adapter activation.
- [x] Add adapter integration tests.

## 18. KV Cache Integration

- [x] Declare layer count for KV cache.
- [x] Declare head count.
- [x] Declare KV head count.
- [x] Declare head dimension.
- [x] Declare cache dtype requirements.
- [x] Declare layout preferences.
- [x] Declare paged cache support.
- [x] Declare append semantics.
- [x] Declare position behavior.
- [x] Preserve Runtime-owned KV cache lifecycle.
- [x] Add KV cache integration tests.

## 19. Prefix Cache Integration

- [ ] Expose architecture metadata for prefix fingerprinting.
- [ ] Include Model Component compatibility in prefix matching where relevant.
- [ ] Preserve Runtime-owned Prefix Cache policy.
- [ ] Add Prefix Cache integration tests.

## 20. Tokenizer Integration

- [x] Declare vocabulary size expectations.
- [x] Declare special token expectations.
- [x] Declare tokenizer family requirements where relevant.
- [x] Declare chat template compatibility where relevant.
- [x] Declare added token behavior where relevant.
- [x] Preserve Tokenizer Contract ownership of encode/decode.
- [x] Add tokenizer integration tests.

## 21. Quantization Integration

- [x] Declare supported quantization method.
- [x] Declare tensor grouping metadata.
- [x] Declare scale metadata requirements.
- [x] Declare zero-point metadata requirements.
- [x] Declare packed layout expectations.
- [x] Declare dequantization Operator requirements.
- [x] Declare quantized Operator requirements.
- [x] Preserve Runtime Kernel selection ownership.
- [x] Add quantization tests.

## 22. Browser Compatibility

- [x] Keep Model Component contract platform-neutral.
- [x] Support WebAssembly Component path.
- [x] Support Runtime-native browser-compatible path.
- [x] Support JavaScript-mediated placeholder.
- [x] Support test fixture path.
- [x] Avoid Wasmtime requirement on browser.
- [x] Avoid native Provider loading requirement.
- [x] Return browser-feature-unsupported where needed.
- [ ] Add wasm32 check where feasible.

## 23. Versioning

- [x] Validate Component Artifact format version.
- [x] Validate Model Component contract version.
- [x] Validate Model Artifact schema version.
- [x] Validate Runtime Capability versions.
- [x] Validate Operator catalog version.
- [x] Validate Execution Graph contract version.
- [x] Validate Adapter contract version where relevant.
- [x] Validate Tokenizer contract version where relevant.
- [x] Add version negotiation tests.

## 24. Conformance

- [x] Define Model Component conformance profile.
- [x] Test architecture metadata validation.
- [x] Test graph production.
- [x] Test Operator requirements.
- [x] Test target module exposure.
- [x] Test adapter compatibility metadata.
- [x] Test KV cache metadata.
- [x] Test tokenizer compatibility metadata.
- [x] Test authority restrictions.
- [x] Test graph validation failure behavior.
- [x] Test no Provider/Device handle exposure.
- [x] Test browser-compatible behavior where applicable.
- [ ] Add conformance report.

## 25. Error Model

- [x] Define model-component-not-found error.
- [x] Define model-component-invalid error.
- [x] Define model-component-untrusted error.
- [x] Define model-component-unsupported-version error.
- [x] Define architecture-unsupported error.
- [x] Define architecture-metadata-invalid error.
- [x] Define model-config-invalid error.
- [x] Define model-artifact-incompatible error.
- [x] Define tokenizer-incompatible error.
- [x] Define operator-catalog-incompatible error.
- [x] Define graph-contract-incompatible error.
- [x] Define graph-production-failed error.
- [x] Define graph-validation-failed error.
- [x] Define target-module-unavailable error.
- [x] Define adapter-incompatible error.
- [x] Define KV-cache-metadata-invalid error.
- [x] Define quantization-unsupported error.
- [x] Define capability-unavailable error.
- [x] Define authority-denied error.
- [x] Define Provider-access-denied error.
- [x] Define Device-access-denied error.
- [x] Define browser-feature-unsupported error.
- [x] Define internal-model-component error.

## 26. Observability

- [x] Emit model component registered observation.
- [x] Emit model component validated observation.
- [x] Emit model component rejected observation.
- [x] Emit architecture compatibility checked observation.
- [x] Emit model config validation failed observation.
- [x] Emit graph production requested observation.
- [x] Emit graph produced observation.
- [x] Emit graph production failed observation.
- [x] Emit target modules exposed observation.
- [x] Emit adapter metadata exposed observation.
- [x] Emit KV cache metadata exposed observation.
- [x] Emit operator requirements declared observation.
- [x] Emit authority denied observation.
- [x] Emit Component-to-Provider access denied observation.
- [x] Emit model component conformance result observation.
- [x] Avoid raw prompt/weight/adapter/cache/handle logging.

## 27. Tests

- [x] Test Model Component identity.
- [x] Test trusted Model Component registration.
- [x] Test untrusted Model Component rejection.
- [x] Test unsupported version rejection.
- [x] Test architecture compatibility success.
- [x] Test architecture unsupported.
- [x] Test invalid config.
- [x] Test target module exposure.
- [x] Test graph production success.
- [x] Test produced graph validation failure.
- [x] Test Operator requirements use portable Operator IDs.
- [x] Test Provider-specific Kernel name rejected as authoritative requirement.
- [x] Test filesystem authority denied.
- [x] Test network authority denied.
- [x] Test Provider handle access denied.
- [x] Test Device handle access denied.
- [x] Test adapter compatibility metadata.
- [x] Test KV cache metadata.
- [x] Test tokenizer compatibility.
- [x] Test quantization metadata.
- [x] Test browser unsupported feature.
- [x] Test raw handles not exposed.

## 28. Documentation

- [x] Document Model Component Contract.
- [x] Document Model Component versus Provider.
- [x] Document Model Component versus Kernel.
- [x] Document Model Component versus Model Artifact.
- [x] Document Model Component identity.
- [x] Document architecture compatibility.
- [x] Document target modules.
- [x] Document graph production.
- [x] Document Operator requirements.
- [x] Document Capability requirements.
- [x] Document authority model.
- [x] Document Provider boundary.
- [x] Document Model Loading relationship.
- [x] Document Model Instance relationship.
- [x] Document Generation relationship.
- [x] Document Adapter relationship.
- [x] Document KV Cache relationship.
- [x] Document Tokenizer relationship.
- [x] Document Quantization relationship.
- [x] Document browser compatibility.
- [x] Document non-goals.

## 29. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Model Component tests.
- [x] Run Component Runtime tests.
- [x] Run Model Artifact tests.
- [x] Run Model Loading tests.
- [x] Run Model Instance tests.
- [x] Run Execution Graph tests.
- [x] Run Operator tests.
- [x] Run Adapter tests.
- [x] Run KV Cache tests.
- [x] Run Tokenizer tests.
- [x] Run OpenSpec validation.
- [x] Run coverage validation.
- [x] Verify Model Component is not Provider.
- [x] Verify Model Component is not Kernel.
- [x] Verify Component-produced graphs are validated.
- [x] Verify no forbidden authority is granted.
- [x] Verify no raw Provider/Device/Kernel handles are exposed.