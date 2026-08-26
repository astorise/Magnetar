# Define Post-Baseline Provider Roadmap

## Why

The first Magnetar baseline is intentionally CPU-only and correctness-first.

That baseline proves:

- Runtime Inference API
- Model Loading
- Model Instance lifecycle
- Qwen-like Model Component baseline
- Execution Graph validation
- Operator semantics
- Kernel Registry and Dispatch
- Tensor Resource and Layout
- Memory Manager
- Reference CPU Provider
- E2E local inference conformance

After the baseline, Magnetar needs a roadmap for optimized and hardware-specific
Providers.

The roadmap must prevent optimized Providers from redefining architecture
contracts or bypassing Runtime validation.

This change defines how post-baseline Providers are introduced.

## What Changes

This change defines Provider roadmap phases after the Reference CPU baseline.

Post-baseline Provider work MAY include:

- optimized CPU Provider
- CUDA Provider
- Metal Provider
- OpenVINO Provider
- QNN Provider
- WebGPU Provider
- quantized execution
- paged attention
- flash attention
- fused kernels
- layout-specialized kernels
- provider-specific conformance
- performance benchmarking

Each Provider SHALL integrate through existing Provider, Kernel, Tensor, Memory,
Operator, Runtime, and Conformance contracts.

## Provider Roadmap Principle

Optimized Providers SHALL improve execution without redefining portable
semantics.

They SHALL NOT introduce model-family Providers such as:

```text
QwenProvider
LlamaProvider
GemmaProvider
```

They SHALL advertise capabilities and Kernels for portable Operators.

## Reference CPU Remains Correctness Baseline

Reference CPU Provider SHALL remain the correctness baseline.

Optimized Providers MAY be compared against Reference CPU outputs for small
fixtures and declared tolerance profiles.

If optimized Provider output differs beyond tolerance, conformance SHALL fail.

## Provider Introduction Phases

Provider introduction SHOULD follow these phases:

```text
1. optimized CPU Provider
2. CUDA Provider
3. Metal Provider
4. OpenVINO Provider
5. QNN Provider
6. WebGPU Provider
7. quantized execution expansion
8. advanced attention kernels
9. production performance profiles
```

The exact order MAY vary by platform priorities, but each Provider must pass
the required conformance gates before being considered production-ready.

## Optimized CPU Provider

An optimized CPU Provider MAY use:

- SIMD
- BLAS
- thread pools
- cache-aware kernels
- fused kernels
- optimized matmul
- optimized normalization
- optimized attention

It SHALL still obey:

- Provider Contract
- Kernel Contract
- Tensor Resource Contract
- Memory Manager accounting
- Kernel Registry selection
- Runtime policy
- conformance tolerances

Optimized CPU Provider SHALL NOT replace Reference CPU as correctness baseline.

## CUDA Provider

CUDA Provider MAY provide GPU execution through CUDA-compatible Devices.

CUDA Provider MAY support:

- device memory
- pinned host memory
- CUDA streams
- CUDA kernels
- cuBLAS or equivalent matmul
- fused kernels
- flash attention where implemented
- quantized kernels where implemented

CUDA Provider SHALL keep native handles internal.

Runtime SHALL not expose CUDA stream, device pointer, module, event, or kernel
handles through public APIs.

## Metal Provider

Metal Provider MAY provide Apple GPU execution.

Metal Provider MAY support:

- Metal buffers
- command queues
- compute pipelines
- MPS or custom kernels
- optimized matmul
- fused kernels
- Apple-specific memory behavior

Metal Provider SHALL keep native handles internal.

Runtime SHALL not expose Metal device, buffer, command queue, pipeline, or event
handles.

## OpenVINO Provider

OpenVINO Provider MAY provide optimized CPU/GPU/NPU execution through OpenVINO.

OpenVINO Provider MAY support:

- graph compilation
- static or dynamic shape profiles
- optimized inference graphs
- quantized execution where supported
- device-specific execution

OpenVINO Provider SHALL still map execution to portable Operator or graph
fragment semantics.

OpenVINO compiled graph internals SHALL remain opaque.

## QNN Provider

QNN Provider MAY support Qualcomm NPU/DSP/GPU execution.

QNN Provider MAY support:

- mobile inference
- NPU-compatible kernels
- quantized execution
- static shape compilation
- provider-owned opaque resources

QNN Provider SHALL not expose native QNN handles.

QNN Provider SHALL clearly report unsupported dynamic behavior.

## WebGPU Provider

WebGPU Provider MAY support browser and native WebGPU execution.

WebGPU Provider MAY support:

- browser-compatible buffers
- WGSL kernels
- WebGPU command submission
- reduced dtype set
- reduced layout set
- browser memory constraints

WebGPU Provider SHALL be compatible with browser constraints and SHALL not
require Wasmtime or native Provider loading.

## Kernel Fusion

Post-baseline Providers MAY implement fused kernels.

Fused kernels SHALL declare semantic equivalence to portable Operator sequences
or graph fragments.

Fusion SHALL NOT change observable inference semantics unless explicitly
declared and validated as an alternative Operator variant.

## Advanced Attention

Post-baseline Providers MAY implement:

- flash attention
- paged attention
- sliding window attention
- block sparse attention
- GQA/MQA optimized attention
- KV-cache-aware attention

Each advanced attention path SHALL declare:

- supported Operator variant
- tensor layout requirements
- memory class requirements
- KV cache layout support
- dtype support
- precision tolerance
- determinism metadata
- fallback behavior

Unsupported advanced attention SHALL fail explicitly.

## Quantized Execution

Post-baseline Providers MAY implement quantized execution.

Quantized support SHALL declare:

- quantization method
- storage dtype
- compute dtype
- accumulation dtype
- scale metadata
- zero-point metadata
- group size
- packing layout
- dequantization behavior
- supported Operators
- conformance tolerance

No hidden quantization or dequantization SHALL occur.

## Layout Expansion

Post-baseline Providers MAY introduce specialized layouts.

Examples:

```text
blocked
paged
packed-quantized
attention-specific
provider-owned-opaque
webgpu-buffer
```

Each layout SHALL be represented through Tensor Layout metadata.

Unsupported layouts SHALL fail or require explicit conversion.

## Memory Expansion

Post-baseline Providers MAY introduce additional memory classes.

Examples:

```text
device
pinned-host
unified
shared
provider-owned
browser-linear-memory
future-webgpu-buffer
```

Memory Manager SHALL track residency, Resource Affinity, transfer, conversion,
and cleanup.

## Provider Conformance Profiles

Each Provider SHALL have conformance profiles.

Profiles MAY include:

```text
provider-core
provider-compute
provider-data-movement
provider-cancellation
provider-observability
provider-dynamic-abi
provider-quantized
provider-advanced-attention
provider-fused-kernel
provider-browser
```

Provider registration SHALL not imply production readiness.

## Performance Benchmarks

Post-baseline work MAY introduce benchmarks.

Benchmarks SHALL be separate from correctness conformance.

Performance benchmarks MAY measure:

- prefill latency
- decode latency
- tokens per second
- memory footprint
- batching throughput
- cache hit behavior
- transfer overhead
- kernel dispatch overhead

A Provider SHALL not pass correctness merely because it is faster.

## Fallback Policy

Fallback across Providers SHALL remain explicit and policy-controlled.

Examples:

```text
CUDA -> optimized CPU
CUDA -> Reference CPU
Metal -> Reference CPU
OpenVINO -> Reference CPU
WebGPU -> browser CPU-like path
```

Fallback SHALL not violate Resource Affinity, privacy, memory, dtype, layout, or
precision policy.

## Runtime API Stability

Post-baseline Providers SHALL not require changes to Runtime Inference API for
basic inference.

New Provider capabilities MAY appear in diagnostics and policy options.

Provider-specific handles SHALL remain hidden.

## CLI Boundary Stability

Provider roadmap SHALL not move Provider-specific authority into
`magnetar-cli`.

`magnetar-cli` may display redacted Provider diagnostics and allow user-facing
policy preferences.

Runtime still owns Provider selection.

## Error Model

Provider roadmap errors SHALL be structured.

Error categories SHOULD include:

- provider roadmap unsupported
- optimized cpu provider unavailable
- cuda provider unavailable
- metal provider unavailable
- openvino provider unavailable
- qnn provider unavailable
- webgpu provider unavailable
- provider feature unsupported
- provider layout unsupported
- provider dtype unsupported
- provider memory class unsupported
- provider advanced attention unsupported
- provider quantization unsupported
- provider fusion invalid
- provider conformance failed
- provider benchmark failed
- provider fallback denied
- provider native handle exposure denied
- internal provider roadmap error

## Observability

Runtime SHOULD emit observations for:

- Provider roadmap feature discovered
- Provider capability advertised
- Provider capability rejected
- optimized Provider selected
- advanced attention selected
- quantized kernel selected
- fused kernel selected
- fallback considered
- fallback used
- fallback denied
- Provider conformance passed
- Provider conformance failed
- benchmark executed
- benchmark skipped

Observability SHALL not expose raw tensor values, raw model weights, raw prompts,
raw KV cache contents, native Provider handles, Device handles, Kernel handles,
memory pointers, secrets, or filesystem paths by default.

## Non-Goals

This change does not:

- implement CUDA
- implement Metal
- implement OpenVINO
- implement QNN
- implement WebGPU
- implement optimized CPU kernels
- implement quantized execution
- implement flash attention
- implement paged attention
- define benchmark numbers
- redefine Operator semantics
- redefine Runtime Inference API
- introduce model-family Providers
- expose native handles

## Impact

Magnetar gains a controlled roadmap beyond the CPU baseline.

The post-baseline direction becomes:

```text
Reference CPU correctness
    |
    v
Optimized Providers
    |
    v
Provider-specific kernels/layouts/memory
    |
    v
Conformance + benchmarks
```

without breaking the inference architecture.