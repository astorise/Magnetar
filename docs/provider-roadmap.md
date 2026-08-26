# Post-Baseline Provider Roadmap

The first Magnetar baseline is CPU-only and correctness-first: Reference CPU
Provider, driven entirely through the Runtime Inference API, Model Loading,
Model Instance lifecycle, Execution Graph validation, Operator semantics,
Kernel Registry and Dispatch, Tensor Resource and Layout, and the Memory
Manager. This roadmap defines how Magnetar introduces optimized and
hardware-specific Providers on top of that baseline without redefining any of
those contracts or letting a Provider bypass Runtime validation.

This document, and the `magnetar-runtime::provider_roadmap` module it
describes, do **not** implement CUDA, Metal, OpenVINO, QNN, WebGPU, optimized
CPU kernels, quantized numerics, flash/paged attention, or benchmark numbers.
They define the roadmap **contract** -- phases, metadata, structured errors,
observability categories, and conformance checks -- that any future
hardware-specific or optimized Provider work must satisfy.

## Provider Roadmap Principle

Optimized Providers improve execution without redefining portable semantics.
They never introduce model-family Providers (`QwenProvider`, `LlamaProvider`,
`GemmaProvider`-shaped names): every Provider identity advertises capabilities
and Kernels, not model-family ownership.
`reject_model_family_provider_name` implements this as an executable,
regression-proof check rather than a documentation-only rule.

Reference CPU Provider remains the correctness baseline. Optimized Provider
output may be compared against Reference CPU fixtures within a declared
tolerance profile; if it differs beyond tolerance, conformance fails.

## Roadmap Phases

`ProviderRoadmapPhase` enumerates the nine introduction phases from the
proposal, in `SHOULD`-order (exact order may vary by platform priorities):

1. optimized CPU Provider
2. CUDA Provider
3. Metal Provider
4. OpenVINO Provider
5. QNN Provider
6. WebGPU Provider
7. quantized execution expansion
8. advanced attention kernels
9. production performance profiles

Every phase's `required_conformance_gates()` includes `provider-core` at
minimum, plus the profile(s) specific to that phase (for example `Cuda` +
`provider-data-movement` for the CUDA phase, `provider-quantized` for the
quantized execution expansion phase). A phase is not production-ready until
its declared gates pass.

## Provider Feature Metadata

`ProviderRoadmapFeature` names every optional (`MAY`, never `SHALL`) feature
called out in the proposal, tagged with the roadmap phase it belongs to:
`is_optional()` is `true` for every variant, and no conformance gate in this
module ever treats one as mandatory. `provider_roadmap_features_for_phase`
looks features up by phase (for example the optimized CPU phase's SIMD,
BLAS, thread-pool execution, cache-aware kernels, and fused kernels; the CUDA
phase's device memory, pinned host memory, streams, kernels, cuBLAS-style
matmul, fused kernels, flash attention, and quantized kernels; and the
Metal/OpenVINO/QNN/WebGPU equivalents).

## Optimized CPU Provider

An optimized CPU Provider may use SIMD, BLAS, thread pools, cache-aware
kernels, and fused kernels -- all optional (`MAY`), never required. It still
obeys the Provider, Kernel, Tensor Resource, and Memory Manager contracts,
Kernel Registry selection, Runtime policy, and conformance tolerances, and it
never replaces Reference CPU as the correctness baseline.

## Hardware Provider Families

`ProviderRoadmapHardwareFamily` (Cuda, Metal, OpenVino, Qnn, WebGpu) captures
the per-family scope in one place instead of five parallel structs:

- **Device metadata**: `device_metadata_template()` builds a representative
  [`DeviceMetadata`] for the family (vendor, architecture, memory class
  support), reusing the existing generic Device contract rather than a
  family-specific type.
- **Memory classes**: `memory_classes()` returns the `KernelMemoryClass`
  subset each family uses (for example CUDA: `Device` + `PinnedHost`; WebGPU:
  `BrowserLinearMemory` + `FutureWebGpuBuffer`).
- **Native handle boundary**: `native_handle_kinds()` names the native
  resources Runtime must never expose (CUDA streams, device pointers,
  modules, events, kernel handles; Metal buffers, command queues, pipelines,
  events; the OpenVINO compiled graph; the QNN native handle).
  `reject_native_handle_exposure` denies every one of them unconditionally.
  WebGPU has no native-handle boundary of its own -- it runs under browser
  sandboxing constraints instead, and `requires_no_native_provider_loading()`
  is `true` only for WebGPU (it must not require Wasmtime or native Provider
  loading).
- **Conformance profile**: `conformance_profile()` maps the family to its
  `ProviderConformanceProfile` variant (`Cuda`, `Metal`, `OpenVino`, `Qnn`,
  `WebGpu`).
- **Fallback**: `primary_fallback_edge()` names the family's default fallback
  target (see "Fallback Policy" below).

OpenVINO additionally keeps compiled graph internals opaque while mapping
execution to portable Operator/graph-fragment semantics. QNN additionally
reports unsupported dynamic behavior explicitly rather than silently
degrading.

## Kernel Fusion

Fused Kernels declare semantic equivalence to a portable Operator sequence or
graph fragment (`KernelFusionMetadata`), a precision tolerance
(`KernelPrecisionMetadata`), and explicit fallback behavior
(`KernelFallbackClass`). `validate_fused_kernel_declaration` rejects a fused
Kernel that is missing any of these, or that does not preserve graph
semantics -- fusion never changes observable inference semantics unless
declared and validated as an alternative Operator variant.

## Advanced Attention

`AdvancedAttentionVariant` names flash attention, paged attention, sliding
window attention, block sparse attention, GQA/MQA-optimized attention, and
KV-cache-aware attention. Each implemented variant must declare (reusing
existing Kernel-contract metadata rather than a new parallel struct):

- the supported Operator variant,
- tensor layout requirements (`TensorLayoutKind`),
- memory class requirements (`KernelMemoryClass`),
- KV cache layout support (`KernelKvCacheMetadata`, required for
  paged/KV-cache-aware variants),
- dtype support (`ComputeDType`),
- precision tolerance (`KernelPrecisionMetadata`),
- determinism metadata (`KernelDeterminism`), and
- fallback behavior (`KernelFallbackClass`).

`validate_advanced_attention_declaration` checks all of the above.
`reject_unsupported_advanced_attention` covers the case where a variant is
not implemented at all: unsupported advanced attention fails explicitly
rather than silently falling back to unvalidated behavior.

## Quantized Execution

`KernelQuantizationMetadata` (defined in `kernel.rs`, alongside the existing
`KernelPrecisionMetadata`/`KernelFusionMetadata`) declares quantization
method, storage/compute/accumulation dtype, scale and zero-point dtype, group
size, packing layout, and dequantization behavior as mandatory fields --
these cannot be omitted by construction. `validate_quantization_declaration`
additionally requires at least one supported Operator and a non-empty
conformance tolerance profile. `reject_hidden_dequantization` makes "no
hidden quantization or dequantization SHALL occur" executable: dequantization
is rejected unless the caller attests it was explicitly declared in the graph
plan.

## Layout Expansion

Blocked, paged, packed-quantized, attention-specific, provider-owned-opaque,
and browser-compatible (WebGPU buffer) layouts are all already representable
through the baseline `TensorLayoutKind` enum -- `POST_BASELINE_LAYOUTS` lists
them. No parallel layout type is introduced.
`require_explicit_layout_conversion` fails a layout mismatch unless an
explicit conversion is declared.

## Memory Expansion

Device, pinned-host, unified, shared, provider-owned, browser-linear-memory,
and future-WebGPU-buffer memory classes are all already representable
through the baseline `KernelMemoryClass` enum -- `POST_BASELINE_MEMORY_CLASSES`
lists all seven. No parallel memory-class type is introduced.
`require_memory_manager_tracking` fails a memory class that is not tracked
for residency, Resource Affinity, transfer, conversion, and cleanup.

## Provider Conformance Profiles

Beyond the baseline `provider-core` / `provider-compute` /
`provider-data-movement` / `provider-cancellation` / `provider-observability`
/ `provider-dynamic-abi` profiles, this roadmap adds:

```text
provider-hardware-cuda       (Cuda)
provider-hardware-metal      (Metal)
provider-hardware-openvino   (OpenVino)
provider-hardware-qnn        (Qnn)
provider-hardware-webgpu     (WebGpu)
provider-quantized           (Quantized)
provider-advanced-attention  (AdvancedAttention)
provider-fused-kernel        (FusedKernel)
provider-browser             (Browser)
```

All nine are optional and `required_by_default() == false`:
`provider_roadmap_conformance_profile_ids` reports them alongside their
required-by-default flag. Provider registration never implies production
readiness or that any of these profiles passed.

## Performance Benchmarks

`ProviderRoadmapBenchmarkCategory` names prefill latency, decode latency,
tokens per second, memory footprint, batching throughput, cache hit
behavior, transfer overhead, and kernel dispatch overhead. This roadmap does
not define benchmark numbers.

Benchmarks are kept structurally separate from conformance:
`ProviderRoadmapBenchmarkResult` is never accepted as input to any
conformance-pass decision in `provider_roadmap.rs` --
`ProviderRoadmapConformanceReport::is_conformant` only ever reads
`ProviderRoadmapConformanceResult`s. A Provider does not pass correctness
merely because it is faster.

## Fallback Policy

`ProviderRoadmapFallbackEdge` names the roadmap's fallback edges:

```text
optimized CPU -> Reference CPU
CUDA -> optimized CPU
CUDA -> Reference CPU
Metal -> Reference CPU
OpenVINO -> Reference CPU
QNN -> Reference CPU
WebGPU -> browser-CPU-like path
```

`evaluate_provider_roadmap_fallback` composes the existing
`reference_cpu::evaluate_fallback` (Resource Affinity, dtype/layout
conversion policy) with roadmap-specific memory, privacy, and precision
policy gates. Every gate in `ProviderRoadmapFallbackContext` defaults to
denying fallback (`deny_by_default`): fallback is explicit and
policy-controlled, never silent, and never violates Resource Affinity,
privacy, memory, dtype, layout, or precision policy.
`evaluate_provider_roadmap_fallback_observed` additionally emits "fallback
considered" then "fallback used"/"fallback denied" observations.

## Runtime API and CLI Boundary Stability

Post-baseline Providers do not require changes to the Runtime Inference API
for basic inference, and Provider-specific handles stay hidden.
`reject_provider_specific_handle_capability` denies capability/scope names
shaped like a Provider-native handle (CUDA stream, Metal buffer, OpenVINO
compiled graph, QNN native handle, ...) while ordinary inference scopes (for
example `"generation"`) remain accepted by the existing
`validate_inference_scope`.

`magnetar-cli` may display redacted Provider diagnostics
(`cli_redacted_provider_diagnostic`, reusing the existing
`redact_backend_diagnostic`) and pass a non-authoritative Provider policy
preference (`ProviderRoadmapPolicyPreference`,
`cli_may_pass_policy_preference`). It cannot select a raw Provider handle
(`reject_cli_raw_provider_handle_selection`). Runtime still owns Provider
selection through Kernel Registry, Resource Affinity, Memory Manager,
readiness, and policy.

## Error Model

`ProviderRoadmapError` covers every error category from the proposal:
`provider-roadmap-unsupported`, the five `*-provider-unavailable` categories
(optimized-cpu/cuda/metal/openvino/qnn/webgpu), `provider-feature-unsupported`,
`provider-layout-unsupported`, `provider-dtype-unsupported`,
`provider-memory-class-unsupported`,
`provider-advanced-attention-unsupported`,
`provider-quantization-unsupported`, `provider-fusion-invalid`,
`provider-conformance-failed`, `provider-benchmark-failed`,
`provider-fallback-denied`, `provider-native-handle-exposure-denied`, and
`internal-provider-roadmap-error`.

## Observability

`ProviderRoadmapObservationKind` covers all fourteen categories from the
proposal (roadmap feature discovered, capability advertised/rejected,
optimized Provider selected, advanced attention selected, quantized kernel
selected, fused kernel selected, fallback considered/used/denied, Provider
conformance passed/failed, benchmark executed/skipped).
`ProviderRoadmapObservation` carries only an observation kind, an optional
Provider name, and a `redacted_metadata` string map whose values are always
passed through `redact_backend_diagnostic` before being stored -- there is no
field through which a raw tensor value, model weight, prompt, KV cache
content, native handle, memory pointer, secret, or filesystem path could
reach the observation.

## Conformance Report

`run_provider_roadmap_conformance` produces a `ProviderRoadmapConformanceReport`
(mirroring `CliBoundaryConformanceReport`) asserting: model-family names are
rejected while hardware/optimized names are allowed; native handle exposure
is denied for every hardware family's handle kinds; fused kernels require a
complete semantic-equivalence declaration; quantized paths require explicit
metadata and reject hidden dequantization; unsupported advanced attention
fails explicitly; fallback is denied by default; the Runtime API surface
rejects Provider-specific handle capabilities while still accepting ordinary
inference scopes; the new conformance profiles are declared without being
required by default; and a fast-but-incorrect result never passes
conformance.

## Local Commands

Run the Provider roadmap tests:

```powershell
cargo test -p magnetar-runtime provider_roadmap -- --nocapture
```

Run the full Runtime suite:

```powershell
cargo test --workspace --all-targets
```

Validate the OpenSpec change:

```powershell
openspec validate define-post-baseline-provider-roadmap --strict
```

## Compatibility Versioning

The current roadmap contract version is `0.1.0`, exposed as
`PROVIDER_ROADMAP_VERSION`. Passing this contract's conformance checks does
not imply any hardware-specific Provider has been implemented -- it only
confirms the roadmap's structural guarantees (no model-family Providers, no
native handle exposure, explicit fusion/quantization/advanced-attention
declarations, deny-by-default fallback, benchmark/conformance separation)
hold in this Runtime revision.
