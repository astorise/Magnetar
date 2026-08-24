# Define Model Component Contract

## Why

Magnetar now defines the full execution substrate:

- Model Artifact
- Model Loading
- Model Instance
- Adapter Loading
- Execution Graph
- Operator Contract
- Kernel Contract
- Kernel Registry and Dispatch
- Provider
- Device
- Memory Manager
- Generation
- Sampling
- KV Cache
- Prefix Cache
- Continuous Batching
- Component Runtime

The missing bridge is the Model Component.

A Model Component represents the portable model architecture implementation.

For example:

```text
qwen component
llama component
gemma component
mistral component
test model component
```

A Model Component is responsible for understanding model architecture metadata
and producing or describing Runtime-valid execution graphs.

It is not a Provider.

It is not a Kernel.

It does not execute CUDA, Metal, OpenVINO, QNN, or CPU kernels directly.

Without a Model Component Contract, architecture-specific logic may leak into:

- Providers
- Kernels
- Model Loading
- Generation
- Adapter Loading
- Scheduler
- Runtime internals

That would reintroduce incorrect abstractions such as:

```text
QwenProvider
LlamaProvider
GemmaProvider
```

This change defines the Model Component boundary.

## What Changes

This change introduces Model Component as a first-class inference Component role.

A Model Component SHALL implement or describe model architecture behavior through
Runtime-authorized inference contracts.

A Model Component MAY:

- validate architecture-specific model metadata
- validate config compatibility
- expose architecture features
- expose target modules
- produce Execution Graphs
- declare required Operators
- declare required Capabilities
- declare KV cache requirements
- declare adapter compatibility
- declare tokenizer compatibility requirements
- declare generation defaults compatibility
- provide shape planning metadata
- provide warmup graph metadata
- provide conformance fixtures

A Model Component SHALL NOT:

- select Providers directly
- select Devices directly
- call Kernels directly
- call native Provider APIs directly
- access arbitrary filesystem
- access network
- access Git
- access secrets
- access process execution
- own workspace state
- own agent/tool orchestration

## Model Component

A Model Component SHALL be a portable Component Artifact with the Model Component
role.

It may be implemented as:

- WebAssembly Component
- Runtime-native architecture implementation
- test fixture implementation
- future browser-compatible component implementation

If implemented as WebAssembly Component, it SHALL use Component Runtime and
authorized WIT contracts.

## Model Component Is Not Provider

A Model Component describes architecture semantics.

A Provider executes Operators and Kernels.

Invalid:

```text
qwen model -> QwenProvider
llama model -> LlamaProvider
```

Correct:

```text
qwen Model Artifact
    + qwen Model Component
    + Execution Graph
    + Operator Contract
    + Kernel Registry
    + CPU/CUDA/Metal/OpenVINO/QNN Provider
```

## Model Component Is Not Kernel

A Model Component may produce an Execution Graph containing Operators.

Kernel selection happens later through Runtime Kernel Registry and Dispatch.

A Model Component SHALL NOT name raw native kernels or function pointers.

## Model Component Is Not Model Artifact

Model Artifact is data.

Model Component is executable or declarative architecture logic.

A Model Artifact may require a compatible Model Component.

A Model Component may support many Model Artifacts of the same architecture
family.

## Model Component Identity

Model Component identity SHALL be stable.

Identity SHOULD include:

- component ID
- component version
- supported architecture families
- supported architecture revisions
- supported Model Artifact schema versions
- supported Runtime Capability versions
- supported Operator catalog version
- supported graph contract version
- trust status
- provenance
- signature state where applicable

## Architecture Compatibility

Model Component SHALL validate architecture compatibility.

Compatibility SHOULD include:

- architecture family
- model type
- hidden size
- layer count
- attention head count
- KV head count
- head dimension
- intermediate size
- vocabulary size
- context length
- position encoding
- normalization kind
- activation kind
- attention variant
- quantization metadata
- tokenizer compatibility
- adapter target modules

Unsupported architecture metadata SHALL produce structured errors.

## Model Config Validation

Model Component MAY validate model config.

Config validation SHALL not trust arbitrary model files blindly.

Runtime SHALL provide validated Model Artifact metadata and authorized config
data.

The Component SHALL not read arbitrary filesystem paths.

## Target Modules

Model Component SHALL expose target module metadata where adapters are supported.

Target modules MAY include:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
lm_head
embedding
norm
attention
mlp
```

Target module names are architecture-specific.

Runtime uses target module metadata for Adapter Loading and graph planning.

## Graph Production

Model Component MAY produce Execution Graphs.

Graph production MAY support phases:

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

Graphs produced by a Model Component SHALL be validated by Runtime before
planning or execution.

## Operator Requirements

Model Component SHALL declare required Operators or Operator families.

Examples:

```text
matmul
embedding
rmsnorm
rope
attention
softmax
silu
add
mul
dtype-conversion
layout-conversion
dequantize
```

Operator declarations SHALL reference portable Operator identities.

They SHALL NOT reference Provider-specific Kernel names as authoritative
requirements.

## Capability Requirements

Model Component SHALL declare required Runtime Capabilities.

Initial Capability areas MAY include:

- model metadata validation
- graph production
- operator catalog access
- tensor descriptor creation
- KV cache metadata
- adapter metadata
- tokenizer metadata
- generation defaults validation
- diagnostics
- observability emit

Capabilities SHALL be inference-scoped.

## Authority

Model Component authority SHALL be inference-scoped.

Allowed authority MAY include:

```text
model-artifact-read
tokenizer-artifact-read
prompt-template-read
adapter-artifact-read
quantization-artifact-read
inference-session-state
generation-session-state
kv-cache-access
prefix-cache-access
compute-capability
generation-capability
sampling-capability
observability-emit
runtime-diagnostics
graph-production
operator-catalog-read
```

Forbidden authority includes:

```text
filesystem
network
env
process
shell
secrets
workspace
git
source-control
tool-execution
external-service
```

## Provider Boundary

Model Component SHALL NOT receive raw Provider handles, Device handles, Kernel
handles, memory pointers, or Provider-owned opaque resources.

Provider and Device information may appear only as Runtime-produced diagnostics
or compatibility metadata where policy allows.

## Graph Validation Boundary

Model Component-produced graphs SHALL be treated as untrusted until validated by
Runtime.

Runtime SHALL validate:

- graph schema
- graph version
- graph phase
- Operator identities
- Operator attributes
- tensor edges
- shape rules
- dtype rules
- layout rules
- Resource Affinity
- memory behavior
- adapter metadata
- KV cache metadata
- policy constraints

## Model Loading Relationship

Model Loading may use Model Component for architecture compatibility and graph
metadata.

Model Loading SHALL not let Model Component bypass artifact trust or memory
admission.

Model Component may participate in:

- config validation
- architecture feature extraction
- target module declaration
- graph metadata preparation
- warmup graph construction

## Model Instance Relationship

A Model Instance may reference the Model Component or architecture
implementation used to create and execute it.

Model Instance identity and lifecycle remain Runtime-owned.

Model Component SHALL not own Model Instance lifecycle.

## Generation Relationship

Generation may request prefill and decode graphs from the Model Component or
architecture implementation.

Generation semantics remain owned by the Generation Contract.

The Model Component SHALL not own:

- request lifecycle
- sampling decisions
- stop conditions
- streaming delivery
- session policy

## Adapter Relationship

Model Component SHALL expose metadata needed to validate adapters.

Adapter-specific graph changes SHALL be explicit.

A Model Component may support:

- adapter target module validation
- adapter overlay graph production
- adapter merge graph production
- provider-fused adapter metadata
- adapter compatibility diagnostics

Adapter activation remains Runtime-owned.

## KV Cache Relationship

Model Component SHALL declare KV cache requirements where relevant.

Metadata SHOULD include:

- layer count
- head count
- KV head count
- head dimension
- cache dtype requirements
- layout preferences
- paged cache support
- append semantics
- position behavior

KV cache allocation and lifecycle remain Runtime-owned.

## Prefix Cache Relationship

Model Component may provide metadata needed for prefix cache compatibility.

Prefix Cache ownership remains Runtime-owned.

Prefix fingerprints may include architecture metadata exposed by the Model
Component.

## Tokenizer Relationship

Model Component SHALL declare tokenizer compatibility requirements.

It may validate:

- vocabulary size
- special token expectations
- tokenizer family where relevant
- chat template compatibility where relevant
- added token behavior where relevant

Tokenizer execution remains owned by Tokenizer Contract.

## Quantization Relationship

Model Component SHALL declare quantization compatibility where relevant.

It may validate:

- quantization method
- tensor grouping
- scale metadata
- zero-point metadata
- packed layout expectations
- dequantization operator requirements
- quantized operator requirements

Provider and Kernel support are resolved later by Runtime.

## Browser Target

Model Component Contract SHALL be platform-neutral.

Browser targets may support Model Components through:

- WebAssembly Component path
- Runtime-native browser-compatible path
- JavaScript-mediated path
- test fixture path

Browser Model Components SHALL not require Wasmtime or native Provider loading.

Unsupported browser features SHALL return structured errors.

## Versioning

Model Component compatibility SHALL be versioned.

Version checks SHOULD include:

- Component Artifact format version
- Model Component contract version
- Model Artifact schema version
- Runtime Capability versions
- Operator catalog version
- Execution Graph contract version
- Adapter contract version where relevant
- Tokenizer contract version where relevant

Breaking changes SHALL require explicit version negotiation or rejection.

## Conformance

Model Components SHALL be subject to conformance tests.

Conformance SHOULD validate:

- architecture metadata validation
- graph production
- Operator requirements
- target module exposure
- adapter compatibility metadata
- KV cache metadata
- tokenizer compatibility metadata
- authority restrictions
- graph validation failure behavior
- no Provider/Device handle exposure
- browser-compatible behavior where applicable

## Error Model

Model Component errors SHALL be structured.

Error categories SHOULD include:

- model component not found
- model component invalid
- model component untrusted
- model component unsupported version
- architecture unsupported
- architecture metadata invalid
- model config invalid
- model artifact incompatible
- tokenizer incompatible
- operator catalog incompatible
- graph contract incompatible
- graph production failed
- graph validation failed
- target module unavailable
- adapter incompatible
- KV cache metadata invalid
- quantization unsupported
- capability unavailable
- authority denied
- Provider access denied
- Device access denied
- browser feature unsupported
- internal model component error

## Observability

Runtime SHOULD emit observations for:

- model component registered
- model component validated
- model component rejected
- architecture compatibility checked
- model config validation failed
- graph production requested
- graph produced
- graph production failed
- target modules exposed
- adapter metadata exposed
- KV cache metadata exposed
- operator requirements declared
- authority denied
- Component-to-Provider access denied
- model component conformance result

Observability SHALL not expose raw model weights, raw prompts, raw adapter
tensors, raw KV cache contents, raw Provider handles, raw Device handles, raw
Kernel handles, or memory pointers by default.

## Non-Goals

This change does not:

- implement Qwen, Llama, Gemma, or Mistral Components
- define concrete model architecture math in full detail
- define Provider Kernel ABI
- define CUDA/Metal/OpenVINO/QNN kernels
- define graph optimizer
- define training
- define fine-tuning
- define model download protocol
- define tool calling
- define agent orchestration
- define workspace/filesystem/network authority
- expose raw Provider handles
- expose raw Kernel handles
- require Wasmtime on browser
- require GPU hardware

## Impact

Magnetar gains the missing architecture layer.

The inference stack becomes:

```text
Model Artifact
    |
    v
Model Component / Architecture Implementation
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
Provider Kernel
    |
    v
Device execution
```

This prepares:

- first concrete Model Components
- first CPU Provider kernels
- architecture-specific test fixtures
- Qwen/Llama/Gemma support without creating QwenProvider/LlamaProvider