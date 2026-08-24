# Tasks

> Progress reflects `magnetar-runtime/src/reference_cpu.rs`, its tests in
> `magnetar-runtime/src/tests.rs`, and the architectural decisions recorded in
> `design.md`.

## 1. Reference CPU Scope

- [x] Define Reference CPU Provider as correctness baseline.
- [x] Document Reference CPU Provider versus optimized CPU Provider.
- [x] Document Reference CPU Provider versus CUDA/Metal/OpenVINO/QNN Providers.
- [x] Document Reference CPU Provider versus Model Component.
- [x] Document Reference CPU Provider fallback policy.
- [x] Document non-goals.

## 2. CPU Provider Module

- [x] Create `provider_cpu` or `reference_cpu` module.
- [x] Export Reference CPU Provider type behind appropriate feature if needed. (No feature gate: always available, matching every other Provider-shaped module in this crate.)
- [x] Keep implementation platform-neutral where feasible. (Builds clean on `wasm32-unknown-unknown`.)
- [x] Avoid requiring external GPU hardware.
- [x] Add module-level documentation.

## 3. Provider Identity

- [x] Define stable Reference CPU Provider ID.
- [x] Define provider kind.
- [x] Define provider version.
- [x] Define built-in status. (`REFERENCE_CPU_BUILT_IN` constant; no shared `ProviderMetadata` field exists for this, so it's expressed as a Reference-CPU-local fact.)
- [x] Define supported Runtime version range. (`REFERENCE_CPU_SUPPORTED_RUNTIME_VERSION_RANGE`, similarly Reference-CPU-local since `ProviderMetadata` only carries a single `api_version`.)
- [x] Define conformance profile status.
- [x] Define feature flags.
- [x] Add provider identity tests.

## 4. CPU Device Metadata

- [x] Expose at least one CPU Device.
- [x] Define CPU Device ID.
- [x] Define CPU Device kind.
- [x] Define host architecture metadata.
- [x] Define logical CPU count metadata where available.
- [x] Define SIMD feature metadata where available. (Runtime-detected `sse4.2`/`avx2`/`fma` on x86_64, expressed as `magnetar:cpu-feature/*` execution capabilities; empty on other architectures.)
- [x] Define memory class support. (`DeviceMetadata.memory_class_support`, a new field added alongside dtype/layout support and execution limits — populated with `{Host}`.)
- [x] Define dtype support. (`DeviceMetadata.dtype_support`, populated with `{Float32}`.)
- [x] Define layout support. (`DeviceMetadata.layout_support`, populated with `{Contiguous}`.)
- [x] Define execution limits. (`DeviceMetadata.execution_limits: DeviceExecutionLimits`, populated from `compute_units` and a workspace byte bound.)
- [x] Define readiness. (Default `Available` via `Device::availability()`.)
- [x] Define pressure. (`DeviceMetadata.pressure: ProviderPressureLevel`, populated with `Low`.)
- [x] Add CPU Device tests.

## 5. Provider Status

- [x] Report lifecycle state. (Default status snapshot derived from health.)
- [x] Report health.
- [x] Report readiness.
- [x] Report admission.
- [x] Report pressure. (`ReferenceCpuProvider::report_pressure`/`status_snapshot` override — pressure is explicitly settable rather than a hardcoded default, since Reference CPU has no automatic load model of its own.)
- [x] Report diagnostics. (Empty by default; no diagnostics generated yet.)
- [x] Integrate with Provider status snapshot.
- [x] Add provider status tests.

## 6. Kernel Advertisement

- [x] Advertise only implemented kernels.
- [x] Advertise implemented Operator IDs.
- [x] Advertise Operator version compatibility.
- [x] Advertise supported dtypes.
- [x] Advertise supported layouts.
- [x] Advertise supported shapes. (Rank-2 constraint for kernels whose tensors are uniformly rank-2 — matmul/rmsnorm/rope/attention/softmax; left unconstrained where ranks legitimately mix, e.g. embedding's rank-1 ids against its rank-2 table.)
- [x] Advertise supported memory classes.
- [x] Advertise workspace requirements. (`KernelWorkspaceRequirements::none()` for most kernels; `attention` requires a Host workspace and requests it through the Memory Manager.)
- [x] Advertise execution mode.
- [x] Advertise determinism metadata.
- [x] Advertise precision metadata.
- [x] Advertise conformance status.
- [x] Add advertisement tests.

## 7. Initial Kernel Set

- [x] Add matmul reference kernel.
- [x] Add embedding reference kernel.
- [x] Add RMSNorm reference kernel.
- [x] Add RoPE reference kernel.
- [x] Add attention reference kernel.
- [x] Add softmax reference kernel.
- [x] Add SiLU reference kernel.
- [x] Add add reference kernel.
- [x] Add mul reference kernel.
- [x] Add residual-add reference kernel.
- [x] Add dtype-conversion reference kernel.
- [x] Add layout-conversion reference kernel.
- [x] Add placeholders for unsupported kernels. (Quantize/dequantize return an explicit error and are never advertised.)
- [x] Add kernel presence tests.

## 8. Matmul Baseline

- [x] Implement correctness-first matmul semantics.
- [x] Validate input ranks.
- [x] Validate shape compatibility.
- [x] Validate dtype support. (Enforced once at the invocation-validation boundary via `KernelAdvertisement::validate_invocation`, not duplicated inside `matmul()` itself.)
- [x] Validate accumulation dtype. (An explicit `accumulation_dtype` attribute other than `f32` is rejected with a structured dtype-unsupported error.)
- [x] Validate output layout. (Enforced at the invocation-validation boundary — only `Contiguous` is advertised/accepted.)
- [x] Use Memory Manager output references. (`ReferenceCpuExecutor::execute_invocation_with_memory_manager` requests allocation and records `TensorResidency` through `MemoryManager`.)
- [x] Add known-output tests.

## 9. Embedding Baseline

- [x] Implement embedding lookup semantics.
- [x] Validate token ID dtype. (Rejects non-integer token ids.)
- [x] Validate token ID bounds.
- [x] Validate embedding table shape.
- [x] Validate output shape.
- [x] Add known-output tests.
- [x] Add out-of-range tests.

## 10. RMSNorm Baseline

- [x] Implement RMSNorm semantics.
- [x] Validate epsilon.
- [x] Validate normalized dimension. (Weight width must match input row width.)
- [x] Validate dtype support. (Enforced at the invocation-validation boundary.)
- [x] Use f32 accumulation where applicable.
- [x] Add known-output tests.
- [x] Add dtype rejection tests.

## 11. RoPE Baseline

- [x] Implement RoPE semantics or first valid placeholder. (Real rotation, not a placeholder.)
- [x] Validate base. (Finite and positive.)
- [x] Validate scale. (Finite and positive.)
- [x] Validate dimension.
- [x] Validate position index mode. (The shared `rope` Operator's attribute schema defines an optional `position_mode` string; Reference CPU implements the default `"sequential"` mode and explicitly rejects any other value.)
- [x] Validate tensor shape.
- [x] Add known-output tests or pending fixture tests.

## 12. Attention Baseline

- [x] Implement simple attention semantics.
- [x] Support causal mode.
- [x] Support simple mask. (The shared `attention_mask_kind` string attribute supports `"causal"`/`"bidirectional"`, validated for consistency with the `causal` boolean; an arbitrary mask *tensor* input was not added, since the shared Operator schema fixes attention arity at 3 inputs (q/k/v) and a 4th input would be a cross-cutting Operator catalog change beyond this Provider.)
- [x] Validate head count.
- [x] Validate KV head count. (Grouped-query attention: `kv_head_count` must evenly divide `head_count`; mismatches are rejected explicitly.)
- [x] Validate head dimension.
- [x] Validate sequence length. (Via shape checks.)
- [x] Validate context length. (`window_size` bounds each query to its most recent keys — sliding-window attention.)
- [x] Use softmax.
- [x] Support KV cache metadata where available. (`KernelKvCacheMetadata` is explicitly present with all support flags `false`, rather than the field being silently absent — no incremental/paged KV cache is implemented, and that is now a stated fact, not an omission.)
- [x] Reject paged attention if unsupported. (Not advertised; `paged-attention` is a distinct Operator this Provider never claims.)
- [x] Add known-output tests. (Including grouped-query, sliding-window, mask-kind, and workspace cases.)

## 13. Softmax Baseline

- [x] Implement softmax semantics.
- [x] Validate axis. (Rank-2 row-wise softmax only; other ranks rejected.)
- [x] Validate dtype. (Enforced at the invocation-validation boundary.)
- [x] Use numerically stable baseline where feasible.
- [x] Add known-output tests.
- [x] Add invalid axis tests.

## 14. Activation Baseline

- [x] Implement SiLU.
- [x] Add GELU placeholder or implementation. (Real tanh-approximation GELU, not a placeholder.)
- [x] Validate activation kind. (A generic `activation` Kernel dispatches on the shared Operator's required `kind` string attribute — `"silu"`/`"gelu"` reuse the same functions as the dedicated kernels; any other kind is rejected explicitly.)
- [x] Validate dtype support. (Enforced at the invocation-validation boundary.)
- [x] Add activation tests.

## 15. Elementwise Baseline

- [x] Implement add.
- [x] Implement mul.
- [x] Implement residual-add.
- [x] Validate broadcasting policy. (Policy is "no broadcasting, exact shape match only," enforced explicitly and tested.)
- [x] Validate dtype compatibility. (Enforced at the invocation-validation boundary.)
- [x] Validate shape compatibility.
- [x] Add elementwise tests.

## 16. DType Conversion Baseline

- [x] Define supported conversions. (Only `f32 -> f32` identity.)
- [x] Reject unsupported conversions.
- [x] Avoid silent conversion.
- [x] Record conversion in graph plan. (Already handled by `KernelDispatchPlan.conversion_steps` at the Runtime layer; no Provider-side action needed.)
- [x] Add conversion tests.

## 17. Layout Conversion Baseline

- [x] Support contiguous identity conversion.
- [x] Define strided placeholder. (Strided targets get a distinct "defined placeholder, not yet implemented" rejection, separate from the generic unsupported-layout rejection.)
- [x] Reject blocked/paged/opaque unsupported layouts.
- [x] Avoid silent layout conversion.
- [x] Add layout conversion tests.

## 18. Quantization Placeholder

- [x] Define unsupported quantization behavior.
- [x] Define dequantize placeholder.
- [x] Reject unsupported quantized formats.
- [x] Add quantization placeholder tests.

## 19. Memory Manager Integration

- [x] Use Runtime resource references.
- [x] Request output allocation through Memory Manager. (`execute_invocation_with_memory_manager` calls `MemoryManager::allocate`.)
- [x] Request workspace through Memory Manager where required. (`ReferenceCpuExecutor::allocate_workspace`, used by `attention`, the one Kernel that advertises a required workspace.)
- [x] Track output readiness. (Via `KernelResult.output_readiness`.)
- [x] Track output residency. (Via `MemoryManager::record_tensor_residency`.)
- [x] Track Resource Affinity. (Recorded on both the allocation request and the `TensorResidency`.)
- [x] Avoid untracked Runtime-visible allocation. (All Runtime-visible resources are `TensorResourceId`s; no raw pointers cross the boundary.)
- [x] Add memory integration tests. (Output-tracking success path, an allocation-rejection/feasibility-failure path, and attention's workspace request/rejection paths.)

## 20. Kernel Registry Integration

- [x] Register kernel advertisements through Kernel Registry. (Also happens automatically via `Runtime::builder().build()`, which calls `Provider::kernel_advertisements()`.)
- [x] Validate advertisements.
- [x] Make kernels discoverable by Operator ID.
- [x] Reject invalid advertisements.
- [x] Add registry integration tests.

## 21. Dispatch Integration

- [x] Execute only Runtime-created Kernel Invocations. (Enforced by `KernelAdvertisement::validate_invocation`.)
- [x] Validate input resource references.
- [x] Validate output resource references.
- [x] Validate workspace references. (`attention` now requires a workspace; the shared `validate_invocation` rejects invocations missing one, and dedicated tests cover both the rejection and the successful Memory-Manager-backed path.)
- [x] Map execution results to Kernel Dispatch Result.
- [x] Map failures to structured errors.
- [x] Add dispatch tests.

## 22. Scheduler Integration

- [x] Expose synchronous execution mode.
- [x] Expose batching support only where implemented. (No `Batched` execution mode is ever advertised, since none is implemented.)
- [x] Expose cancellation support metadata. (`KernelCancellationSupport::TimeoutOnly` — synchronous execution can't cooperatively cancel mid-kernel, but an already-elapsed deadline is honored before dispatch starts.)
- [x] Expose timeout behavior. (`invocation.deadline_millis == Some(0)` is treated as an already-elapsed deadline and fails fast with `KernelError::KernelTimeout`.)
- [x] Add scheduler metadata tests.

## 23. Fallback Policy

- [x] Allow CPU fallback only when policy permits.
- [x] Make CPU fallback observable. (`ReferenceCpuExecutor::evaluate_fallback_observed` emits considered/used/failed Kernel observations.)
- [x] Reject fallback when Resource Affinity forbids host movement.
- [x] Reject fallback when dtype/layout policy forbids conversion. (`FallbackPolicyContext` models required-vs-allowed dtype/layout conversion explicitly.)
- [x] Add fallback tests.

## 24. Conformance

- [x] Define Reference CPU conformance profile.
- [x] Use Reference CPU for Operator semantic tests.
- [x] Produce reference outputs for small fixtures.
- [x] Validate dtype behavior. (Via this Provider's own dtype-conversion tests, not the legacy `ProviderCompute`/`ComputeGraph` conformance profile, which is a separate, older advertisement system this Provider deliberately does not participate in — see `design.md`.)
- [x] Validate layout behavior. (Via this Provider's own layout-conversion tests, same rationale.)
- [x] Validate memory behavior. (Via this Provider's own Memory Manager integration tests, same rationale.)
- [x] Validate error behavior.
- [x] Add conformance report. (`ReferenceCpuExecutor::run_conformance_checks` — a small, fixed set of known-input/known-output checks against `matmul`/`rmsnorm`/`softmax`/`silu`/`add`, returning a `ReferenceCpuConformanceReport` and emitting a `KernelConformanceResult` observation.)

## 25. Browser Compatibility

- [x] Keep contract platform-neutral.
- [x] Do not require browser implementation.
- [x] Avoid native loading assumption in browser.
- [x] Return browser unsupported errors where needed. (`attention` is explicitly rejected with `reference-cpu-browser-feature-unsupported` on `wasm32`, since its Host-class Memory-Manager-backed workspace requirement is not meaningful against browser linear memory; not advertised as `browser_compatible`.)
- [x] Add wasm32 check where feasible.

## 26. Error Model

- [x] Define reference-cpu-provider-unavailable error.
- [x] Define reference-cpu-provider-disabled-by-policy error.
- [x] Define reference-cpu-device-unavailable error.
- [x] Define reference-cpu-kernel-not-found error.
- [x] Define reference-cpu-dtype-unsupported error.
- [x] Define reference-cpu-layout-unsupported error.
- [x] Define reference-cpu-shape-unsupported error.
- [x] Define reference-cpu-memory-class-unsupported error.
- [x] Define reference-cpu-workspace-unavailable error.
- [x] Define reference-cpu-execution-failed error.
- [x] Define reference-cpu-deterministic-mode-unsupported error.
- [x] Define reference-cpu-precision-unsupported error.
- [x] Define reference-cpu-conformance-failed error.
- [x] Define reference-cpu-fallback-denied error.
- [x] Define reference-cpu-browser-feature-unsupported error.
- [x] Define internal-reference-cpu error.

## 27. Observability

- [x] Emit Reference CPU Provider registered observation. (`ReferenceCpuProvider::initialize` records a `ProviderRegistered` Kernel observation — a new `KernelObservationKind` variant, purely additive to the shared enum, on this Provider's own executor, since no Runtime-wide provider-registration observability channel exists yet for any Provider.)
- [x] Emit Reference CPU Device detected observation. (Same mechanism, `DeviceDetected` variant.)
- [x] Emit Reference CPU Kernel advertised observation. (Emitted generically by `KernelRegistry::register_provider_advertisement` for any Provider; exercised by this Provider's registry test.)
- [x] Emit Reference CPU Kernel selected observation. (Emitted generically by `KernelRegistry::select`; exercised by a dedicated test.)
- [x] Emit Reference CPU dispatch started observation.
- [x] Emit Reference CPU dispatch completed observation.
- [x] Emit Reference CPU dispatch failed observation.
- [x] Emit Reference CPU fallback considered observation.
- [x] Emit Reference CPU fallback used observation.
- [x] Emit Reference CPU conformance result observation. (`run_conformance_checks` emits `KernelObservationKind::KernelConformanceResult`, an existing shared variant.)
- [x] Emit unsupported dtype/layout/shape observation. (The dispatch-failed observation already carries the structured error id, which identifies the dtype/layout/shape rejection.)
- [x] Emit memory feasibility failure observation. (`KernelObservationKind::KernelMemoryFeasibilityFailed`, emitted when `MemoryManager::allocate` rejects the request.)
- [x] Avoid raw tensor/prompt/weight/cache/handle logging.

## 28. Tests

- [x] Test Reference CPU Provider registration. (Via the conformance-suite test, which registers into a real `Runtime`.)
- [x] Test CPU Device metadata.
- [x] Test Provider status snapshot.
- [x] Test kernel advertisement validation.
- [x] Test matmul known output.
- [x] Test embedding known output.
- [x] Test RMSNorm known output.
- [x] Test RoPE known output or pending fixture.
- [x] Test attention known output.
- [x] Test softmax known output.
- [x] Test SiLU known output.
- [x] Test add/mul/residual-add known output.
- [x] Test dtype conversion explicitness.
- [x] Test layout conversion explicitness.
- [x] Test unsupported quantization.
- [x] Test Memory Manager output tracking.
- [x] Test Kernel Registry discovery.
- [x] Test Runtime-created invocation required.
- [x] Test CPU fallback denied by default.
- [x] Test CPU fallback allowed by policy.
- [x] Test no raw handles exposed.

## 29. Documentation

- [x] Document Reference CPU Provider. (Module-level rustdoc in `reference_cpu.rs`.)
- [x] Document CPU Device metadata.
- [x] Document initial kernel set.
- [x] Document correctness-first policy.
- [x] Document host memory execution.
- [x] Document dtype support.
- [x] Document layout support.
- [x] Document attention baseline.
- [x] Document RoPE baseline.
- [x] Document RMSNorm baseline.
- [x] Document Memory Manager integration.
- [x] Document Kernel Registry integration.
- [x] Document fallback policy.
- [x] Document conformance role.
- [x] Document browser limitations.
- [x] Document non-goals.

## 30. Final Validation

- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run wasm32 check where feasible.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run Reference CPU Provider tests.
- [x] Run Kernel tests.
- [x] Run Kernel Registry tests.
- [x] Run Operator tests.
- [x] Run Execution Graph tests.
- [x] Run Memory Manager tests.
- [x] Run Provider conformance tests.
- [x] Run OpenSpec validation.
- [x] Run coverage validation. (`cargo llvm-cov`, available via the `llvm-tools-preview` toolchain component: ~79% region / ~74% function coverage on `reference_cpu.rs`.)
- [x] Verify Reference CPU is correctness baseline.
- [x] Verify CPU fallback is explicit.
- [x] Verify unsupported kernels are not assumed.
- [x] Verify raw handles are not exposed.
