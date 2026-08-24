# Define Qwen Model Component Baseline

## Why

Magnetar now has the architecture needed to support real model families without
creating model-family Providers.

The Runtime has:

- Model Artifact Contract
- Model Loading Contract
- Model Instance Lifecycle
- Model Component Contract
- Execution Graph Contract
- Operator Contract
- Kernel Contract
- Kernel Registry and Dispatch
- Tensor Resource and Layout Contract
- Reference CPU Provider baseline
- First Operator Implementation Scope
- Tokenizer Contract
- Generation Contract
- KV Cache and Prefix Cache contracts
- Adapter Loading Contract

The next step is to define the first concrete Model Component baseline.

This change defines a Qwen-like decoder-only Model Component baseline.

The goal is not to implement every Qwen variant.

The goal is to create a clear first architecture target that exercises the
runtime path:

```text
Model Artifact
  -> Model Component
  -> Model Instance
  -> Execution Graph
  -> Operators
  -> Kernel Registry
  -> Reference CPU Provider
  -> Generation
```

## What Changes

This change introduces a Qwen Model Component baseline.

The Qwen Model Component SHALL be an inference-scoped architecture
implementation that can validate Qwen-like model metadata and produce portable
Execution Graphs for first baseline inference.

It SHALL use only Runtime-authorized contracts.

It SHALL NOT be a Provider.

It SHALL NOT execute Kernels directly.

It SHALL NOT access arbitrary filesystem, network, process, Git, secrets, or
workspace state.

The exact Rust type names are implementation-defined.

## Qwen Model Component

A Qwen Model Component SHALL represent Qwen-like decoder-only transformer
architecture behavior.

It MAY be implemented as:

- Runtime-native architecture implementation
- WebAssembly Component
- test fixture implementation
- browser-compatible placeholder

The first implementation MAY start Runtime-native and later move to a WASM
Component path.

## Qwen Component Is Not Qwen Provider

The Qwen Model Component SHALL NOT introduce a `QwenProvider`.

Invalid:

```text
Qwen Model Artifact -> QwenProvider -> Qwen kernels
```

Correct:

```text
Qwen Model Artifact
  + Qwen Model Component
  + portable Execution Graph
  + portable Operators
  + Kernel Registry
  + Reference CPU / CUDA / Metal / OpenVINO / QNN Providers
```

## First Baseline Scope

The Qwen baseline SHALL target the first operator implementation scope.

It SHOULD use required-now operators:

```text
embedding
rmsnorm
matmul
rope
attention
softmax
silu
add
mul
residual-add
dtype-conversion
layout-conversion
```

It SHALL not require:

```text
flash-attention
paged-attention
quantized-matmul
fused-mlp
fused-rmsnorm
Provider-assisted sampling
beam search helpers
speculative decoding helpers
training operators
```

Unsupported requirements SHALL fail explicitly.

## Supported Architecture Class

The first Qwen baseline SHALL target a decoder-only transformer class.

Supported metadata SHOULD include:

- architecture family
- model type
- hidden size
- layer count
- attention head count
- key/value head count
- head dimension
- intermediate size
- vocabulary size
- context length
- RoPE metadata
- RMSNorm epsilon
- activation kind
- tied or untied embeddings metadata
- tensor naming or logical module mapping
- generation default compatibility metadata
- tokenizer compatibility metadata

## Config Validation

The Qwen Model Component SHALL validate model configuration metadata supplied by
Runtime.

Config validation SHALL include:

- architecture family is Qwen-compatible
- model type is decoder-only
- hidden size is valid
- layer count is valid
- attention head count is valid
- KV head count is valid
- head dimension is valid
- hidden size matches head count and head dimension where required
- intermediate size is valid
- vocabulary size is valid
- context length is valid
- activation kind is supported
- normalization kind is supported
- RoPE metadata is supported
- unsupported features are rejected

The Component SHALL not read arbitrary config files by path.

## Model Artifact Compatibility

The Qwen Model Component SHALL validate Model Artifact compatibility.

Compatibility SHOULD include:

- artifact architecture family
- artifact schema version
- weight tensor inventory
- tensor naming convention or logical mapping
- required tensor shapes
- tokenizer association
- generation config compatibility
- quantization metadata
- adapter compatibility metadata
- trust state supplied by Runtime

The Component SHALL not bypass artifact trust validation.

## Tensor Inventory

The baseline SHOULD define expected logical tensors.

Expected logical tensor groups MAY include:

```text
token_embedding
layers.N.input_norm
layers.N.self_attn.q_proj
layers.N.self_attn.k_proj
layers.N.self_attn.v_proj
layers.N.self_attn.o_proj
layers.N.post_attn_norm
layers.N.mlp.gate_proj
layers.N.mlp.up_proj
layers.N.mlp.down_proj
final_norm
lm_head
```

Exact artifact names MAY be mapped to logical names by Runtime-controlled
metadata.

A missing required logical tensor SHALL produce structured errors.

## Target Modules

The Qwen Model Component SHALL expose target modules for adapters.

Initial target modules SHOULD include:

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
```

Target module metadata SHOULD include:

- logical module name
- layer selector
- expected tensor shape
- supported adapter methods
- merge/overlay compatibility
- graph insertion point

## Tokenizer Compatibility

The Qwen Model Component SHALL declare tokenizer compatibility requirements.

Validation SHOULD include:

- vocabulary size compatibility
- special token metadata
- EOS token availability
- BOS token policy where relevant
- pad token policy where relevant
- chat template compatibility metadata where relevant
- added token behavior where relevant

Tokenizer execution remains owned by the Tokenizer Contract.

## Generation Defaults Compatibility

The Qwen Model Component SHALL validate generation default compatibility where
model metadata provides defaults.

It MAY validate:

- maximum context length
- EOS token references
- BOS token policy
- default temperature/top-p/top-k metadata as non-authoritative defaults
- stop token metadata

Generation semantics remain owned by the Generation Contract.

## Prefill Graph

The Qwen Model Component SHALL be able to produce or describe a prefill
Execution Graph for the first baseline.

A prefill graph SHOULD include:

```text
input token IDs
  -> embedding
  -> repeated decoder layers
  -> final rmsnorm
  -> logits matmul
```

Each decoder layer SHOULD be expressible with required-now operators.

Prefill graph SHALL expose KV cache write or append behavior where KV cache is
enabled.

## Decode Graph

The Qwen Model Component SHALL be able to produce or describe a decode Execution
Graph for the first baseline.

A decode graph SHOULD process one or a small number of new tokens.

It SHOULD consume prior KV cache where available.

It SHOULD produce logits for Sampling.

Decode graph SHALL preserve Generation and Sampling boundaries.

## Decoder Layer Graph

The baseline decoder layer SHOULD be expressible as:

```text
hidden
  -> rmsnorm
  -> q_proj matmul
  -> k_proj matmul
  -> v_proj matmul
  -> rope on q/k
  -> attention
  -> o_proj matmul
  -> residual-add
  -> rmsnorm
  -> gate_proj matmul
  -> silu
  -> up_proj matmul
  -> mul
  -> down_proj matmul
  -> residual-add
```

This graph may be unfused.

Fused kernels are not required.

## Attention Baseline

Qwen attention baseline SHALL use portable attention metadata.

Metadata SHOULD include:

- causal attention
- attention mask kind
- query head count
- KV head count
- head dimension
- sequence length
- context length
- RoPE dependency
- KV cache behavior
- dtype requirements
- layout requirements

If grouped-query attention is not implemented by first CPU attention baseline,
Runtime SHALL reject or use a policy-defined compatible path only when correct.

No silent change of attention semantics is allowed.

## RoPE Baseline

Qwen RoPE metadata SHALL be explicit.

It SHOULD include:

- base
- scale
- dimension
- position index mode
- context length compatibility
- dynamic scaling support status

Unsupported RoPE variants SHALL fail explicitly.

## MLP Baseline

Qwen MLP baseline SHOULD use unfused required-now operators:

```text
gate = matmul(normed_hidden, gate_weight)
up = matmul(normed_hidden, up_weight)
activated = silu(gate)
mlp_hidden = mul(activated, up)
out = matmul(mlp_hidden, down_weight)
```

No fused MLP kernel is required.

## Normalization Baseline

Qwen baseline SHOULD use RMSNorm.

RMSNorm metadata SHALL include epsilon and normalized dimension.

LayerNorm is not required for this Qwen baseline.

## Logits Projection

The Qwen baseline SHALL produce logits through matmul against `lm_head` or tied
embedding weights according to model metadata.

Sampling remains separate.

Provider-assisted sampling is not required.

## KV Cache Metadata

The Qwen Model Component SHALL declare KV cache metadata.

Metadata SHOULD include:

- layer count
- KV head count
- attention head count
- head dimension
- cache dtype
- sequence dimension
- batch dimension
- position encoding behavior
- append behavior
- supported layout preference
- paged cache support status

The baseline MAY support non-paged KV cache first.

Paged KV cache may remain placeholder.

## Prefix Cache Compatibility

The Qwen Model Component SHALL expose metadata needed for Prefix Cache
compatibility.

Metadata MAY include:

- architecture family
- component version
- tokenizer compatibility metadata
- RoPE metadata
- active adapter set influence
- model config fingerprint influence

Prefix Cache ownership remains Runtime-owned.

## Adapter Compatibility

The Qwen Model Component SHALL expose adapter compatibility metadata.

It SHOULD support LoRA metadata validation for target modules where possible.

Adapter activation and loading remain Runtime-owned.

If adapter graph modifications are unsupported in the first baseline, Runtime
SHALL reject adapter activation or run adapter-free graph according to policy.

## Quantization Compatibility

The first Qwen baseline MAY reject quantized artifacts unless explicit
dequantization support exists.

Quantization metadata SHALL be validated.

No hidden dequantization SHALL occur.

Quantized execution is not required for this change.

## Tensor Layout Scope

The first Qwen baseline SHOULD target host contiguous tensors through Reference
CPU Provider.

Unsupported tensor layouts SHALL fail explicitly or require explicit conversion.

Provider-owned opaque layouts SHALL not be exposed to the Qwen Model Component.

## DType Scope

The first Qwen baseline SHOULD target f32 compute.

Other storage dtypes may be rejected or explicitly converted according to policy.

No silent dtype conversion SHALL occur.

## Model Instance Relationship

A ready Qwen Model Instance SHALL reference the Qwen Model Component or
Runtime-native architecture implementation used.

Component version and architecture metadata may participate in cache
compatibility.

Model Instance lifecycle remains Runtime-owned.

## Model Loading Relationship

Model Loading SHALL resolve a compatible Qwen Model Component or native
architecture implementation before publishing a ready Qwen Model Instance.

Model Loading MAY use the Qwen Component for:

- architecture validation
- tensor inventory validation
- target module metadata
- graph metadata preparation
- warmup graph preparation

It SHALL not allow Component logic to bypass trust, memory, provider, device, or
policy validation.

## Generation Relationship

Generation SHALL request prefill and decode graph behavior through Runtime.

The Qwen Component may produce graphs, but Generation owns:

- request lifecycle
- prefill/decode orchestration
- stop conditions
- sampling invocation
- streaming
- usage accounting
- cancellation

## Reference CPU Relationship

The Qwen baseline SHALL be executable on Reference CPU only if all required
operators have compatible CPU kernels.

If a required operator is missing, graph planning SHALL fail with a structured
error.

## Component Runtime Relationship

If implemented as a WASM Component, the Qwen Model Component SHALL run through
Component Runtime.

It SHALL receive only inference-scoped authority.

It SHALL not receive filesystem, network, process, shell, secrets, Git,
workspace, Provider handle, Device handle, Kernel handle, or raw tensor pointer
authority.

## Versioning

The Qwen Model Component baseline SHALL declare compatibility versions.

Version metadata SHOULD include:

- Qwen baseline contract version
- Model Component contract version
- Model Artifact schema version
- Operator catalog version
- Execution Graph contract version
- Tensor contract version
- Tokenizer contract version
- KV cache contract version
- Adapter contract version where relevant

Unsupported versions SHALL fail explicitly.

## Conformance

The Qwen Model Component baseline SHALL have conformance fixtures.

Fixtures SHOULD include:

- valid minimal config
- invalid architecture family
- invalid hidden/head configuration
- missing tensor inventory
- invalid tensor shape
- target module exposure
- prefill graph production
- decode graph production
- required operator scope validation
- tokenizer compatibility validation
- KV cache metadata validation
- adapter target validation
- unsupported quantization rejection
- authority denial
- no Provider/Device/Kernel handle exposure

## Error Model

Qwen Model Component errors SHALL be structured.

Error categories SHOULD include:

- qwen component not found
- qwen component invalid
- qwen component untrusted
- qwen component unsupported version
- qwen architecture unsupported
- qwen config invalid
- qwen tensor inventory missing
- qwen tensor shape mismatch
- qwen tokenizer incompatible
- qwen generation metadata invalid
- qwen operator unsupported
- qwen graph production failed
- qwen graph validation failed
- qwen target module unavailable
- qwen adapter unsupported
- qwen KV cache metadata invalid
- qwen RoPE unsupported
- qwen attention variant unsupported
- qwen quantization unsupported
- qwen dtype unsupported
- qwen layout unsupported
- qwen Reference CPU coverage missing
- qwen capability unavailable
- qwen authority denied
- qwen browser feature unsupported
- internal qwen component error

## Observability

Runtime SHOULD emit observations for:

- Qwen Component resolved
- Qwen Component validated
- Qwen Component rejected
- Qwen config validated
- Qwen tensor inventory checked
- Qwen target modules exposed
- Qwen tokenizer compatibility checked
- Qwen KV metadata produced
- Qwen prefill graph produced
- Qwen decode graph produced
- Qwen graph validation failed
- Qwen required operator missing
- Qwen Reference CPU coverage missing
- Qwen authority denied
- Qwen conformance result

Observability SHALL not expose raw prompts, raw model weights, raw adapter
tensors, raw KV cache contents, raw Provider handles, raw Device handles, raw
Kernel handles, or memory pointers by default.

## Browser Target

The Qwen baseline contract SHALL be platform-neutral.

Browser execution is not required for this change.

A browser target may return unsupported if required component, tensor, provider,
or kernel paths are unavailable.

Browser paths SHALL not require Wasmtime or native Provider loading.

## Non-Goals

This change does not:

- implement every Qwen model variant
- guarantee production Qwen performance
- define Hugging Face download behavior
- define full safetensors loading implementation
- define GGUF loading
- define quantized Qwen execution
- define flash attention
- define paged attention implementation
- define CUDA/Metal/OpenVINO/QNN kernels
- define Provider-assisted sampling
- define training or fine-tuning
- define tool calling
- define chat template execution itself
- define CLI behavior
- create QwenProvider
- expose raw model weights or handles

## Impact

Magnetar gains a first concrete architecture baseline.

The initial executable path becomes:

```text
Qwen-like Model Artifact
    |
    v
Qwen Model Component baseline
    |
    v
Prefill / Decode Execution Graph
    |
    v
Required-now Operators
    |
    v
Reference CPU Kernels
    |
    v
Generation logits
```

This prepares:

- Runtime inference API
- magnetar-cli inference boundary
- end-to-end local inference conformance