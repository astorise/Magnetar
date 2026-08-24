# Define Reference CPU Provider And Kernel Baseline

## Why

Magnetar now defines:

- Operator Contract
- Execution Graph
- Kernel Contract
- Kernel Registry and Dispatch
- Provider model
- Device model
- Memory Manager
- Model Component contract
- Model Instance lifecycle

The architecture is ready to define the first concrete execution baseline.

Before adding CUDA, Metal, OpenVINO, QNN, WebGPU, or optimized kernels, Magnetar
needs a Reference CPU Provider.

The Reference CPU Provider establishes correctness.

It provides a slow but predictable implementation path for:

- operator conformance
- graph validation
- kernel dispatch
- Memory Manager integration
- model component testing
- end-to-end local inference
- fallback testing
- browser-independent baseline semantics

Without this baseline, optimized Providers may define behavior before portable
semantics are proven.

## What Changes

This change introduces a Reference CPU Provider.

The Reference CPU Provider SHALL be a built-in or test-enabled Provider that
implements a minimal set of correct CPU kernels for initial inference.

It SHALL provide:

- CPU Provider identity
- CPU Device metadata
- kernel advertisements
- host memory execution path
- deterministic reference behavior where feasible
- correctness-oriented implementations
- conformance fixtures
- structured errors
- observability

The Reference CPU Provider is not required to be fast.

Correctness and clarity are preferred over performance.

## Reference CPU Provider

The Reference CPU Provider SHALL be the baseline Provider for portable operator
semantics.

It MAY be enabled in:

```text
development
test
conformance
local-runtime
fallback-policy
```

Production use is allowed only if Runtime policy permits it.

## Provider Identity

The Reference CPU Provider SHALL expose stable Provider identity.

The identity SHOULD include:

- provider ID
- provider kind
- provider version
- built-in status
- supported Provider ABI version if applicable
- supported Runtime version range
- conformance profile status
- feature flags

Example conceptual provider ID:

```text
magnetar:provider/reference-cpu
```

## CPU Device

The Reference CPU Provider SHALL expose at least one CPU Device.

CPU Device metadata SHOULD include:

- device ID
- device kind
- host architecture
- logical CPU count where available
- SIMD feature metadata where available
- memory class support
- supported dtype metadata
- supported layout metadata
- execution limits
- readiness
- pressure
- diagnostics

CPU Device identity SHALL remain Runtime/Provider-owned and opaque.

## Scope

The Reference CPU Provider SHALL initially target correctness for a minimal
operator set.

Initial operator families SHOULD include:

```text
tensor
linear-algebra
normalization
position-encoding
attention
activation
layout
dtype-conversion
quantization-placeholder
sampling-support-placeholder
```

The exact first implementation subset may be defined by the first operator
implementation scope change.

## Initial Kernel Baseline

The Reference CPU Provider SHOULD advertise baseline kernels for:

```text
matmul
batched-matmul placeholder
embedding
rmsnorm
layernorm placeholder
rope
attention
softmax
silu
gelu placeholder
add
mul
residual-add
dtype-conversion
layout-conversion
dequantize placeholder
sampling-helper placeholder
```

Unsupported kernels SHALL be advertised as unsupported or absent.

Runtime SHALL not assume kernels that are not advertised.

## Correctness Over Performance

Reference CPU kernels SHALL prioritize correctness, debuggability, and
conformance.

They MAY use simple implementations.

They SHALL not rely on hidden Provider-specific semantics.

They SHOULD avoid nondeterministic behavior where feasible.

Performance optimizations MAY be added later only if conformance remains stable.

## Host Memory Execution

The Reference CPU Provider SHALL operate primarily on host memory.

It MAY support:

```text
host
browser-linear-memory placeholder
```

It MAY reject:

```text
device
pinned-host
unified
provider-owned
future-webgpu-buffer
```

unless explicit support is implemented.

## Layout Support

The Reference CPU Provider SHALL support a minimal portable layout set.

Initial layouts SHOULD include:

```text
contiguous
strided placeholder
```

Quantized, blocked, paged, or Provider-owned opaque layouts MAY be unsupported
initially.

Unsupported layouts SHALL produce structured errors or explicit conversion
requirements.

## DType Support

The Reference CPU Provider SHALL declare dtype support explicitly.

Initial supported dtypes MAY include:

```text
f32
f16 placeholder
bf16 placeholder
i32
u32
u8
i8
bool
```

If f16 or bf16 are not natively computed, Runtime SHALL declare whether they are:

- unsupported
- converted to f32 explicitly
- represented as storage only
- used only in test fixtures

No silent dtype conversion SHALL occur.

## Quantization Scope

Quantization support MAY be placeholder-only in this change.

The Reference CPU Provider may support dequantization fixtures or simple
dequantize operations.

Full GGUF, GPTQ, AWQ, NF4, or provider-optimized quantized kernels are non-goals
for this change.

## Attention Baseline

Reference attention SHALL be correctness-oriented.

It SHOULD support:

- causal attention
- simple attention mask
- query/key/value tensors
- softmax
- value aggregation
- optional KV cache integration where defined
- f32 accumulation where applicable

It MAY be slow.

Paged attention MAY be unsupported.

Flash attention is not required.

## RoPE Baseline

Reference RoPE SHALL support explicit position metadata.

It SHOULD validate:

- base
- scale
- dimension
- position index mode
- tensor shape
- dtype compatibility

Dynamic scaling MAY be unsupported initially.

## RMSNorm Baseline

Reference RMSNorm SHALL support explicit epsilon and accumulation behavior.

It SHOULD use f32 accumulation where applicable.

Unsupported dtype combinations SHALL be rejected or converted explicitly.

## Kernel Advertisement

The Reference CPU Provider SHALL advertise kernels using the Kernel Contract.

Advertisements SHALL include:

- implemented Operator
- supported dtypes
- supported layouts
- supported shapes
- memory classes
- workspace requirements
- execution mode
- determinism metadata
- precision metadata
- conformance profile status

## Kernel Registry Integration

Reference CPU kernels SHALL enter execution only through Kernel Registry
validation.

The Provider SHALL not bypass Runtime dispatch.

The Runtime SHALL treat Reference CPU kernels like any other Provider-owned
Kernel.

## Memory Manager Integration

The Reference CPU Provider SHALL use Memory Manager resource references.

CPU kernels SHALL not allocate untracked Runtime-visible memory.

Temporary buffers and outputs SHALL be planned through Memory Manager where they
affect Runtime resource accounting.

Internal stack/local temporary values are allowed only when not visible as
Runtime resources.

## Scheduler Integration

The Reference CPU Provider SHALL expose execution mode metadata usable by the
Scheduler.

Initial execution may be synchronous.

Asynchronous CPU execution MAY be added later.

The Scheduler SHALL not assume GPU-like behavior.

## Fallback Policy

The Reference CPU Provider MAY be used as fallback only when Runtime policy
explicitly allows it.

Fallback to CPU SHALL be explicit and observable.

Fallback SHALL not silently move Device-bound data to host.

Fallback SHALL not violate Resource Affinity, dtype, layout, determinism,
precision, privacy, or memory policy.

## Conformance Role

The Reference CPU Provider SHALL serve as a baseline for conformance.

It MAY be used to:

- validate operator semantics
- compare optimized kernel outputs
- generate reference outputs
- test Kernel Registry selection
- test explicit fallback
- test graph execution correctness

Reference outputs SHALL be deterministic where feasible.

Tolerance profiles SHALL be explicit.

## Test Fixtures

The Reference CPU Provider SHOULD include test fixtures for:

- small tensors
- known matmul outputs
- known normalization outputs
- known RoPE outputs
- known softmax outputs
- known attention outputs
- dtype conversion behavior
- layout conversion behavior
- error cases

Fixtures SHALL avoid reliance on external GPU hardware.

## Browser Target

The Reference CPU Provider contract SHALL be platform-neutral.

A browser-compatible CPU-like Provider MAY exist later.

This change SHALL not require browser support.

Browser targets SHALL not require native CPU Provider loading.

Unsupported browser use SHALL return structured errors.

## Error Model

Reference CPU Provider errors SHALL be structured.

Error categories SHOULD include:

- reference cpu provider unavailable
- reference cpu provider disabled by policy
- reference cpu device unavailable
- reference cpu kernel not found
- reference cpu dtype unsupported
- reference cpu layout unsupported
- reference cpu shape unsupported
- reference cpu memory class unsupported
- reference cpu workspace unavailable
- reference cpu execution failed
- reference cpu deterministic mode unsupported
- reference cpu precision unsupported
- reference cpu conformance failed
- reference cpu fallback denied
- reference cpu browser feature unsupported
- internal reference cpu error

## Observability

Runtime SHOULD emit observations for:

- Reference CPU Provider registered
- Reference CPU Device detected
- Reference CPU Kernel advertised
- Reference CPU Kernel selected
- Reference CPU dispatch started
- Reference CPU dispatch completed
- Reference CPU dispatch failed
- Reference CPU fallback considered
- Reference CPU fallback used
- Reference CPU conformance result
- Reference CPU unsupported dtype/layout/shape
- Reference CPU memory feasibility failure

Observability SHALL not expose raw tensor values, prompts, model weights, KV
cache contents, memory pointers, Provider handles, or Device handles by default.

## Non-Goals

This change does not:

- optimize CPU performance
- require SIMD implementation
- require BLAS integration
- require CUDA
- require Metal
- require OpenVINO
- require QNN
- require WebGPU
- define full quantized inference
- define GGUF loading
- define graph optimizer
- define production-grade CPU serving
- define model architecture Components
- define CLI behavior
- expose raw memory pointers
- allow silent CPU fallback
- require browser implementation

## Impact

Magnetar gains a concrete correctness-first execution baseline.

The first executable stack becomes possible:

```text
Model Component
    |
    v
Execution Graph
    |
    v
Operator Invocation
    |
    v
Kernel Registry
    |
    v
Reference CPU Kernel
    |
    v
Host memory result
```

This prepares:

- first operator implementation scope
- Qwen model component baseline
- Runtime inference API
- end-to-end local inference conformance