# Define Execution Graph And Operator Contract

## Why

Magnetar now has contracts for:

- Model Artifacts
- Model Loading
- Model Instance lifecycle
- Adapter Loading
- Tokenizer
- Generation
- Sampling
- Inference Sessions
- KV Cache
- Prefix Cache
- Continuous Batching
- Memory Manager
- Providers and Devices

Before defining kernels, Magnetar must define what kernels implement.

A kernel is an implementation detail for a Provider and Device.

The portable semantic unit must be the Operator.

The Runtime also needs a representation for model computation as an Execution
Graph.

Without an Execution Graph and Operator Contract, kernels would likely be
defined directly as architecture-specific functions.

That would produce incorrect boundaries such as:

```text
cuda_llama_attention
metal_qwen_mlp
candle_gemma_rmsnorm
```

instead of:

```text
operator: attention
operator: matmul
operator: rmsnorm
operator: rope
provider-specific kernel implementation
```

This change defines the portable semantic layer between Model Components and
Provider kernels.

## What Changes

This change introduces:

- Execution Graph
- Operator
- Operator invocation
- Operator metadata
- Tensor edge metadata
- Shape constraints
- DType constraints
- Layout constraints
- Memory requirements
- Resource Affinity constraints
- Provider Capability requirements
- graph validation
- graph planning boundary
- graph execution boundary
- adapter-aware graph behavior
- KV-cache-aware graph behavior

The exact Rust type names are implementation-defined.

## Operator

An Operator SHALL represent a portable semantic operation.

Examples include:

```text
matmul
batched-matmul
embedding
rmsnorm
layernorm
rope
attention
paged-attention
softmax
activation
gelu
silu
add
mul
residual-add
dtype-conversion
layout-conversion
quantize
dequantize
sampling-helper
logits-processor-helper
```

An Operator describes what is computed.

A Kernel describes how a Provider computes it.

## Operator Is Not Kernel

An Operator SHALL NOT be a Provider-specific kernel.

Invalid examples:

```text
cuda_flash_attention_kernel
metal_qwen_rmsnorm
candle_llama_matmul
```

Correct examples:

```text
operator: attention
operator: rmsnorm
operator: matmul
kernel: Provider-specific implementation selected later
```

## Execution Graph

An Execution Graph SHALL represent a Runtime-understandable composition of
operators and tensor/resource edges.

A graph MAY represent:

- full model forward
- prefill subgraph
- decode subgraph
- attention block
- MLP block
- adapter overlay path
- logits processing helper path
- model warmup graph
- test fixture graph

The graph is not required to be a universal deep learning framework graph.

It is a Magnetar inference execution graph.

## Graph Producer

An Execution Graph MAY be produced by:

- Runtime-native architecture implementation
- Model Component
- Provider-assisted graph builder
- test fixture
- future imported model representation

A graph producer SHALL not bypass Runtime validation.

If produced by a Component, the Component describes or emits graph structure
through authorized inference contracts.

The Component SHALL NOT receive raw Provider handles or Device handles.

## Graph Consumer

Runtime SHALL validate and plan the Execution Graph.

Providers may execute graph fragments or operators through Runtime dispatch.

Scheduler may schedule graph execution phases.

Memory Manager plans allocations and residency.

Kernel Registry later selects concrete kernels.

## Operator Catalog

Magnetar SHALL define an initial operator catalog.

The catalog SHOULD be versioned.

Initial operator families SHOULD include:

```text
tensor
linear-algebra
normalization
position-encoding
attention
activation
quantization
layout
sampling-support
control
```

This change defines the catalog structure and core semantics.

It does not require all operators to be fully optimized.

## Operator Identity

Operator identity SHALL be stable and versioned.

Operator identity SHOULD include:

- namespace
- name
- semantic version
- operator family
- input contract
- output contract
- attribute schema
- shape rules
- dtype rules
- layout rules
- memory behavior
- determinism metadata
- error behavior

Example conceptual IDs:

```text
magnetar:operator/matmul@1
magnetar:operator/attention@1
magnetar:operator/rmsnorm@1
magnetar:operator/rope@1
```

## Operator Attributes

Operators MAY have attributes.

Attributes SHALL be validated.

Examples:

```text
matmul:
  transpose_a
  transpose_b
  accumulation_dtype

rope:
  base
  scale
  dimension
  position_mode

attention:
  causal
  window_size
  head_count
  kv_head_count
  head_dimension
  attention_mask_kind

rmsnorm:
  epsilon

activation:
  kind
```

Attribute values SHALL not select Provider or Device directly.

## Tensor Edges

Execution Graph edges SHALL describe tensor/resource flow.

Tensor edge metadata SHOULD include:

- logical tensor ID
- shape
- dtype
- layout
- memory class
- residency constraints
- Resource Affinity
- mutability
- lifetime hint
- aliasing behavior
- producer operator
- consumer operators

Edges SHALL not expose raw memory pointers.

## Shape Contract

Operators SHALL define shape requirements and shape inference behavior where
possible.

Shape validation SHALL happen before execution when metadata is available.

Dynamic shapes MAY be supported.

Shape errors SHALL be structured.

## DType Contract

Operators SHALL define supported logical dtypes and accumulation dtypes.

DType validation SHALL distinguish:

```text
input dtype
output dtype
storage dtype
compute dtype
accumulation dtype
```

Unsupported dtype combinations SHALL fail before kernel dispatch where possible.

## Layout Contract

Operators SHALL define required or supported tensor layouts.

Layout MAY include:

- contiguous
- strided
- blocked
- paged
- provider-specific opaque
- quantized packed layout
- attention-specific layout
- browser-compatible layout

Provider-specific opaque layout SHALL not leak into portable Component APIs.

Layout conversion SHALL be explicit.

## Memory Behavior

Operators SHALL declare memory behavior.

Memory behavior MAY include:

- reads input
- writes output
- mutates input
- aliases output
- requires workspace
- can operate in-place
- requires host-visible memory
- requires device-resident memory
- requires pinned memory
- supports streaming output
- supports paged KV cache

Memory Manager SHALL use memory behavior for planning.

## Resource Affinity

Operators and tensor edges SHALL preserve Resource Affinity.

If an operator consumes Device-bound data, Runtime SHALL select compatible
execution or insert explicit movement/conversion operations where policy allows.

Silent movement SHALL be forbidden.

## Execution Graph Validation

Runtime SHALL validate Execution Graph before execution.

Validation SHALL include:

- graph identity
- graph version
- operator identities
- operator attributes
- input/output arity
- tensor edge consistency
- shape compatibility
- dtype compatibility
- layout compatibility
- Resource Affinity
- memory behavior
- aliasing rules
- lifecycle/resource constraints
- Provider Capability feasibility
- policy constraints

Invalid graphs SHALL not execute.

## Execution Graph Planning

Runtime SHALL plan Execution Graph execution before Provider submission.

Planning SHALL determine:

- operator execution order
- fusion opportunities placeholder
- memory allocation needs
- workspace needs
- data movement requirements
- layout conversion requirements
- dtype conversion requirements
- KV cache use
- adapter paths
- Provider/Device compatibility
- kernel selection placeholder
- batching compatibility
- failure handling

Kernel selection is finalized by a later Kernel Registry and Dispatch contract.

## Graph Execution

Graph execution SHALL run through Runtime-owned execution paths.

Graph execution SHALL not allow Components to call Providers directly.

Graph execution SHALL not expose raw Provider handles, Device handles, memory
pointers, or raw tensor storage.

## Graph Phases

Graphs MAY be phase-specific.

Initial graph phases SHOULD include:

```text
model-load
warmup
prefill
decode
adapter-activation
adapter-merge
sampling-helper
test
```

A graph phase SHALL define expected inputs, outputs, memory behavior, and
lifecycle constraints.

## Prefill And Decode Graphs

Transformer inference SHOULD distinguish prefill graph from decode graph where
useful.

Prefill graph may process many tokens.

Decode graph may process one or a small number of tokens per active sequence.

Continuous Batching may schedule these phases differently.

## Attention Operator

Attention SHALL be represented as an operator family.

Attention metadata SHOULD include:

- causal mode
- attention mask kind
- query head count
- key/value head count
- head dimension
- sequence length
- context length
- KV cache usage
- paged cache support
- position encoding dependency
- dtype requirements
- layout requirements

Specific kernel implementations are defined later.

## Paged Attention

Paged attention MAY be represented as an attention variant or attribute.

This change SHALL not require paged attention implementation.

It SHALL ensure the Operator Contract can represent paged KV cache metadata.

## RoPE Operator

Rotary position embedding SHALL be represented as a position-encoding operator
or attention attribute according to graph design.

RoPE metadata SHOULD include:

- base
- scale
- dimension
- position index mode
- dynamic scaling where supported
- model compatibility

## Normalization Operators

Normalization operators SHOULD include RMSNorm and LayerNorm families.

Normalization metadata SHOULD include epsilon, normalized dimension, dtype, and
accumulation dtype.

## Quantization Operators

Quantization-related operators MAY include:

- quantize
- dequantize
- requantize
- quantized-matmul
- unpack
- pack
- scale-apply

Quantization metadata SHALL remain explicit.

Quantized operator support SHALL be validated against Provider capabilities.

## Adapter-Aware Graphs

Adapters may modify or extend graphs.

Adapter behavior MAY appear as:

- additional operators
- modified matmul path
- fused adapter attributes
- merge graph
- overlay graph
- provider-fused path

Adapter changes SHALL be explicit.

Adapter graph changes SHALL affect graph identity and cache compatibility where
semantics change.

## KV-Cache-Aware Graphs

Graphs used for Generation MAY consume or produce KV cache references.

KV cache usage SHALL be explicit in graph metadata.

Graph execution SHALL not expose raw KV cache contents.

KV cache compatibility SHALL be validated before graph execution.

## Prefix-Cache-Aware Graphs

Prefix Cache hits may alter prefill graph boundaries.

Graph planning SHALL account for reused prefix length and backing KV cache.

Prefix Cache reuse SHALL not bypass graph validation.

## Sampling Helper Graphs

Sampling remains a separate contract.

However, some sampling helper operations may be represented as graph operators
where Provider-assisted or Device-resident processing is useful.

Sampling semantics remain owned by the Sampling Contract.

## Operator Determinism

Operators SHOULD declare determinism metadata.

Determinism MAY depend on:

- dtype
- Provider
- Device
- kernel implementation
- parallel reduction behavior
- sampling RNG
- memory layout
- browser/native target

Determinism metadata SHALL be surfaced to Generation and Sampling where needed.

## Operator Errors

Operator errors SHALL be structured.

Error categories SHOULD include:

- operator not found
- operator version unsupported
- operator attribute invalid
- input arity invalid
- output arity invalid
- shape mismatch
- shape unsupported
- dtype unsupported
- dtype conversion required
- dtype conversion unsupported
- layout unsupported
- layout conversion required
- layout conversion unsupported
- memory behavior unsupported
- workspace unavailable
- Resource Affinity conflict
- Provider capability unavailable
- kernel unavailable
- graph validation failed
- graph planning failed
- graph execution failed
- browser feature unsupported
- internal operator error

## Observability

Runtime SHOULD emit observations for:

- graph created
- graph validation started
- graph validation failed
- graph validated
- graph planning started
- graph planning completed
- operator planned
- data movement inserted
- dtype conversion inserted
- layout conversion inserted
- workspace requested
- graph execution started
- operator execution started
- operator execution completed
- operator execution failed
- graph execution completed
- graph execution failed

Observability SHALL not expose raw tensor values, raw prompts, raw model
weights, raw KV cache contents, raw Provider handles, or raw memory pointers by
default.

## Browser Target

Execution Graph and Operator contracts SHALL be platform-neutral.

Browser targets may support a reduced operator set depending on:

- WebAssembly linear memory
- future WebGPU buffers
- browser-compatible Provider path
- available Component Engine
- Memory Manager policy

Unsupported operators or layouts SHALL return structured errors.

Browser execution SHALL not require Wasmtime or native Provider loading.

## Non-Goals

This change does not:

- define concrete kernel implementation
- define kernel registry
- define Provider kernel ABI
- require CUDA/Metal/OpenVINO/QNN kernels
- define full graph optimizer
- require operator fusion implementation
- define ONNX import
- define StableHLO import
- define distributed graph execution
- define training operators
- define automatic differentiation
- define browser WebGPU implementation
- expose raw tensor memory
- expose Provider handles
- allow Components to call Providers directly

## Impact

Magnetar gains a portable semantic layer between Model Components and kernels.

The execution stack becomes:

```text
Model Component / architecture implementation
        |
        v
Execution Graph
        |
        v
Operator Contract
        |
        v
Kernel Registry and Dispatch
        |
        v
Provider-specific Kernel
        |
        v
Device execution
```

This prepares:

- Kernel Contract
- Kernel Registry and Dispatch
- Model Component Contract