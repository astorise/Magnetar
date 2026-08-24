# Define Kernel Contract

## Why

Magnetar now has an Execution Graph and Operator Contract.

Operators define portable semantics.

The next layer is the Kernel.

A Kernel is the concrete implementation of an Operator for a Provider and
possibly a Device class.

Without a Kernel Contract, Provider implementations may expose arbitrary
functions with unclear compatibility, shape limits, dtype behavior, workspace
requirements, layout assumptions, memory ownership, determinism, and fallback
semantics.

That would break Runtime planning and make graph execution unsafe.

This change defines the Kernel Contract.

## What Changes

This change introduces Kernel as a first-class execution implementation unit.

A Kernel SHALL implement one Operator or one explicitly declared operator
variant.

A Kernel SHALL declare:

- Kernel identity
- implemented Operator identity and version range
- Provider ownership
- Device compatibility
- supported dtypes
- supported layouts
- supported shapes
- supported memory classes
- input/output aliasing behavior
- workspace requirements
- Resource Affinity constraints
- determinism metadata
- execution mode
- error mapping
- conformance requirements
- fallback behavior

The exact Rust type names are implementation-defined.

## Kernel

A Kernel SHALL be a concrete implementation of an Operator.

Examples:

```text
operator: magnetar:operator/matmul@1
kernel: cuda.matmul.tensorcore.fp16

operator: magnetar:operator/rmsnorm@1
kernel: cpu.rmsnorm.avx2.fp32

operator: magnetar:operator/attention@1
kernel: cuda.attention.flash.v2
```

Kernel names are implementation metadata.

Kernel identity SHALL not replace Operator identity.

## Kernel Is Not Operator

A Kernel SHALL NOT define portable semantics.

Portable semantics remain owned by the Operator Contract.

The Kernel declares that it implements a compatible Operator.

If Kernel behavior differs from the Operator semantics, Runtime SHALL reject it
or treat it as a different operator variant.

## Kernel Is Not Provider

A Kernel belongs to or is exposed by a Provider.

It is not itself a Provider.

Provider owns registration, health, readiness, pressure, Device exposure, and
native execution boundary.

Kernel owns implementation metadata and execution behavior for a specific
operator path.

## Kernel Is Not Model Architecture

Kernels SHALL NOT be named or selected primarily by model family.

Invalid examples:

```text
qwen_kernel
llama_attention_kernel
gemma_mlp_kernel
```

Correct pattern:

```text
operator: attention
kernel: cuda.flash_attention
graph/model metadata: qwen-compatible attention attributes
```

Model architecture may constrain attributes, shapes, layouts, and graph paths,
but it SHALL not turn a kernel into a model-specific Provider abstraction.

## Kernel Identity

Kernel identity SHALL be stable within its Provider.

Kernel identity SHOULD include:

- provider ID
- kernel name
- kernel version
- implemented Operator ID
- implemented Operator version range
- supported feature flags
- implementation family
- optional build fingerprint
- optional conformance profile reference

Kernel identity SHALL not expose raw function pointers.

## Kernel Advertisement

Providers SHALL advertise kernels through Runtime-readable metadata.

Kernel advertisements SHOULD include:

- Kernel identity
- implemented Operator
- supported dtypes
- supported layouts
- supported shapes
- supported memory classes
- supported Devices or Device classes
- workspace requirements
- supported execution modes
- determinism metadata
- precision metadata
- performance hints
- fallback hints
- required Provider features
- required Device features

Runtime SHALL not rely on undocumented kernel behavior.

## Kernel Selection Deferred

This change defines the Kernel Contract.

It does not define the complete Kernel Registry and Dispatch algorithm.

Kernel selection is finalized by a later change.

However, the Kernel Contract SHALL provide all metadata needed by the future
Kernel Registry.

## Operator Compatibility

A Kernel SHALL declare which Operator it implements.

Runtime SHALL validate:

- Operator ID compatibility
- Operator version compatibility
- attribute compatibility
- input/output arity
- shape compatibility
- dtype compatibility
- layout compatibility
- memory behavior compatibility
- determinism compatibility
- Resource Affinity compatibility

A Kernel SHALL not execute an incompatible Operator invocation.

## Shape Constraints

A Kernel SHALL declare shape constraints.

Shape constraints MAY include:

- rank requirements
- static dimension requirements
- dynamic dimension support
- alignment requirements
- batch size limits
- sequence length limits
- head count limits
- head dimension limits
- matrix tile constraints
- block size constraints
- page size constraints
- maximum total elements
- maximum total tokens

Shape constraints SHALL be validated before dispatch where possible.

## DType Constraints

A Kernel SHALL declare dtype constraints.

DType constraints SHOULD include:

- input dtype support
- output dtype support
- compute dtype support
- accumulation dtype support
- storage dtype support where relevant
- quantized dtype support
- mixed precision support
- conversion requirements

Unsupported dtype combinations SHALL fail before execution where possible.

## Layout Constraints

A Kernel SHALL declare layout constraints.

Layout constraints MAY include:

- contiguous
- strided
- blocked
- packed
- quantized packed
- paged KV cache
- attention-specific layout
- Provider-owned opaque layout
- browser-compatible layout

Provider-owned opaque layouts SHALL not leak into portable Component APIs.

If layout conversion is required, Runtime planning SHALL make it explicit.

## Memory Class Constraints

A Kernel SHALL declare supported memory classes.

Examples include:

```text
host
pinned-host
device
unified
shared
provider-owned
browser-linear-memory
future-webgpu-buffer
```

Kernel execution SHALL not require memory classes that were not validated.

## Workspace Requirements

A Kernel SHALL declare workspace requirements.

Workspace metadata SHOULD include:

- required or optional workspace
- size formula or upper bound
- memory class requirement
- alignment
- lifetime
- reuse policy
- per-operation or per-batch scope
- failure behavior when unavailable

Workspace allocation SHALL be performed through Memory Manager.

## Input And Output Aliasing

A Kernel SHALL declare aliasing behavior.

Aliasing metadata MAY include:

- no aliasing allowed
- input-output alias allowed
- in-place supported
- output aliases input
- temporary aliasing internal only
- mutation of input
- read-only input
- write-only output

Runtime SHALL validate aliasing before dispatch.

## Resource Affinity

Kernel execution SHALL preserve Resource Affinity.

A Kernel that requires inputs on a particular Provider or Device SHALL declare
that requirement.

Runtime SHALL not silently move data to satisfy a Kernel.

Movement, copy, layout conversion, dtype conversion, or materialization SHALL be
explicitly planned.

## Execution Mode

A Kernel SHALL declare execution mode.

Initial modes SHOULD include:

```text
synchronous
asynchronous
streamed
batched
graph-captured
provider-fused
browser-compatible
test-fixture
```

Execution mode affects scheduling, cancellation, memory lifetime, and
observability.

## Cancellation

A Kernel SHALL declare cancellation support.

Cancellation support MAY include:

- not-supported
- before-dispatch-only
- cooperative
- interruptible
- timeout-only
- Provider-specific cancellation

Runtime SHALL not assume cancellation is supported.

If cancellation is unsupported during execution, Runtime SHALL report that
limitation.

## Determinism

A Kernel SHALL declare determinism metadata.

Determinism MAY depend on:

- dtype
- Device
- execution mode
- parallel reductions
- accumulation order
- atomic operations
- kernel version
- Provider version
- hardware features
- random state where relevant

When deterministic mode is requested, Runtime SHALL validate Kernel determinism
support or return a structured error.

## Precision And Numerical Behavior

A Kernel SHOULD declare precision metadata.

Metadata MAY include:

- accumulation dtype
- rounding mode where known
- approximate math usage
- fused operation semantics
- allowed tolerance profile
- quantization error profile
- deterministic tolerance profile

Conformance tests SHALL define acceptable tolerances.

## Fused Kernels

A Kernel MAY implement a fused operator path.

Fused kernels SHALL declare the semantic operator or operator group they
implement.

If a fused kernel implements multiple Operators, Runtime SHALL validate that
the fusion preserves graph semantics.

Fusion SHALL not silently change observable inference semantics.

## Provider-Fused Adapter Kernels

A Kernel MAY support adapter-aware fused execution.

Adapter-aware kernels SHALL declare:

- supported adapter methods
- maximum rank
- supported adapter dtypes
- supported merge/overlay strategy
- supported target modules
- active adapter constraints

Adapter-aware kernel execution SHALL validate active adapter compatibility.

## KV-Cache-Aware Kernels

A Kernel MAY consume or produce KV cache.

Such kernels SHALL declare:

- KV cache layout support
- paged cache support
- append behavior
- read behavior
- cache dtype support
- cache memory class support
- Resource Affinity constraints

Raw KV cache contents SHALL not be exposed.

## Prefix-Cache-Aware Kernels

Kernels generally do not own Prefix Cache.

However, prefix reuse may affect kernel input boundaries.

Kernel metadata SHALL be able to support adjusted sequence or context lengths
caused by Prefix Cache reuse.

## Batched Kernels

A Kernel MAY support batched execution.

Batched kernel metadata SHOULD include:

- max batch size
- max active sequences
- max total tokens
- sequence length constraints
- padding behavior
- ragged batch support
- paged KV cache compatibility
- per-operation output mapping
- batch slot compatibility

Continuous Batching SHALL use this metadata.

## Browser-Compatible Kernels

Kernel Contract SHALL be platform-neutral.

Browser-compatible kernels MAY target:

- WebAssembly linear memory
- JavaScript-mediated execution
- future WebGPU buffers
- browser-compatible Provider paths

Browser kernels SHALL not require Wasmtime or native Provider loading.

Unsupported browser kernel features SHALL return structured errors.

## Kernel Invocation

A Kernel Invocation SHALL be Runtime-created.

It SHOULD include:

- invocation ID
- Operator invocation reference
- Kernel identity
- input resource references
- output resource references
- workspace reference
- execution mode
- Provider/Device context metadata
- Resource Affinity metadata
- cancellation token
- deadline or timeout
- observability correlation
- policy metadata

Components SHALL NOT create raw Kernel Invocations directly against Providers.

## Kernel Results

Kernel execution SHALL return structured results.

Results SHOULD include:

- success or failure
- output readiness
- updated resource metadata
- workspace release hints
- timing metadata where available
- determinism metadata
- precision diagnostics where available
- Provider diagnostics
- Device diagnostics
- structured error

Results SHALL not expose raw Provider handles or memory pointers.

## Error Model

Kernel errors SHALL be structured.

Error categories SHOULD include:

- kernel not found
- kernel version unsupported
- kernel Operator mismatch
- kernel attribute unsupported
- kernel shape unsupported
- kernel dtype unsupported
- kernel layout unsupported
- kernel memory class unsupported
- kernel workspace unavailable
- kernel aliasing unsupported
- kernel Resource Affinity conflict
- kernel Device unsupported
- kernel Provider unavailable
- kernel Provider not ready
- kernel Provider saturated
- kernel execution failed
- kernel cancellation unsupported
- kernel cancelled
- kernel timeout
- kernel determinism unsupported
- kernel precision unsupported
- kernel conformance failed
- kernel browser feature unsupported
- internal kernel error

## Conformance

A Kernel SHALL be subject to conformance testing.

Kernel conformance SHOULD validate:

- Operator semantic correctness
- shape handling
- dtype handling
- layout handling
- memory behavior
- aliasing behavior
- workspace behavior
- Resource Affinity behavior
- cancellation behavior where supported
- determinism claims
- precision tolerance
- error mapping
- observability metadata

Conformance SHALL be tied to Operator semantics and Kernel metadata.

## Fallback

Kernel fallback behavior SHALL be explicit.

Fallback may include:

- alternate kernel
- alternate Provider
- alternate Device
- explicit dtype conversion
- explicit layout conversion
- host execution
- rejection

Fallback SHALL not silently violate Resource Affinity, dtype policy, layout
policy, determinism policy, or memory policy.

## Security And Isolation

Kernel execution SHALL not expose raw memory, raw model weights, raw prompts,
raw KV cache contents, raw Provider handles, or raw Device handles to Components
or clients.

Provider kernels are trusted native execution code unless running in a sandboxed
implementation.

The Kernel Contract does not make native kernels safe from arbitrary code
execution risks.

## Observability

Runtime SHOULD emit observations for:

- kernel advertised
- kernel selected placeholder
- kernel invocation created
- kernel dispatch started
- kernel dispatch completed
- kernel dispatch failed
- kernel workspace requested
- kernel cancellation requested
- kernel cancelled
- kernel timeout
- kernel fallback considered
- kernel fallback used
- kernel conformance result
- kernel Resource Affinity conflict
- kernel determinism limitation
- kernel precision diagnostic

Observability SHALL not expose raw tensor values, prompts, weights, KV cache
contents, Provider handles, Device handles, or memory pointers by default.

## Non-Goals

This change does not:

- define Kernel Registry selection algorithm
- define Kernel Dispatch algorithm fully
- define Provider kernel ABI
- implement CUDA kernels
- implement Metal kernels
- implement OpenVINO kernels
- implement QNN kernels
- implement WebGPU kernels
- define graph optimizer
- define fusion optimizer
- define model architecture Components
- expose raw kernel function pointers
- expose Provider handles to Components
- require GPU hardware
- require browser kernel implementation

## Impact

Magnetar gains a stable contract for Provider-specific operator
implementations.

The execution stack becomes:

```text
Execution Graph
        |
        v
Operator Invocation
        |
        v
Kernel Contract
        |
        v
future Kernel Registry / Dispatch
        |
        v
Provider-owned Kernel
        |
        v
Device execution
```

This prepares:

- Kernel Registry and Dispatch
- Provider kernel capability expansion
- Model Component Contract