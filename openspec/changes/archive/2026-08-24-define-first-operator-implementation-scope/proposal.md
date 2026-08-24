# Define First Operator Implementation Scope

## Why

Magnetar now defines:

- Operator Contract
- Execution Graph
- Kernel Contract
- Kernel Registry and Dispatch
- Reference CPU Provider
- Model Component Contract
- Memory Manager
- Generation
- Sampling
- KV Cache
- Prefix Cache

The Operator Catalog can describe many possible operations.

But the first implementation must be deliberately small.

If Magnetar tries to implement all operators at once, the first executable path
will become unclear.

This change defines the first operator implementation scope.

The goal is to identify the minimum required operator set for a first
correctness-oriented local inference path using the Reference CPU Provider.

## What Changes

This change classifies operators into implementation tiers.

Initial tiers SHALL include:

```text
required-now
required-for-first-decoder-model
placeholder
explicitly-unsupported
future-optimized
```

The first implementation scope SHALL target a minimal decoder-only transformer
path.

The scope SHALL support enough operators to validate:

- graph production
- graph validation
- graph planning
- Kernel Registry selection
- Reference CPU dispatch
- Memory Manager output tracking
- basic prefill/decode execution
- conformance fixtures

## First Target

The first implementation target SHOULD be a small decoder-only transformer
baseline.

It does not need to support a full production model immediately.

It SHOULD be sufficient for:

```text
embedding
rmsnorm
matmul
rope
attention
softmax
silu
elementwise add/mul
residual add
logits projection
```

This target prepares later Qwen/Llama/Gemma support.

## Required-Now Operators

The first required operator set SHOULD include:

```text
embedding
matmul
rmsnorm
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

These operators SHALL have explicit semantics, validation, Reference CPU kernel
coverage, and conformance fixtures.

## Required For First Decoder Model

Operators required for a first decoder-only model SHOULD include:

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
```

The implementation MAY use simple unfused paths.

Fused kernels are not required.

## Placeholder Operators

The following operators MAY exist as placeholders:

```text
batched-matmul
layernorm
gelu
dequantize
quantize
requantize
quantized-matmul
paged-attention
sampling-helper
logits-processor-helper
layout-pack
layout-unpack
```

Placeholder means:

- identity is reserved
- metadata shape may exist
- Runtime can report unsupported
- conformance can mark pending
- Kernel Registry must not assume implementation

## Explicitly Unsupported Operators

Operators outside the first scope SHALL fail explicitly.

Unsupported behavior SHALL not be silently emulated unless policy defines a
validated fallback.

Examples that may be unsupported initially:

```text
paged-attention
flash-attention
quantized-matmul
grouped quantization kernels
moe-dispatch
speculative-decoding helpers
beam-search helpers
training operators
gradient operators
```

## Future Optimized Operators

Some operators may later gain optimized kernels.

Examples:

```text
flash-attention
fused-rmsnorm
fused-mlp
fused-rope-attention
tensorcore-matmul
simd-rmsnorm
paged-attention
quantized-matmul
```

Future optimization SHALL not change portable Operator semantics.

## First CPU Kernel Coverage

The Reference CPU Provider SHALL advertise required-now kernels only when they
are implemented.

For required-now operators, Reference CPU Provider SHOULD provide correctness
kernels.

For placeholders, it SHALL either omit the kernel advertisement or advertise
unsupported metadata where the Kernel Contract allows it.

## No Silent Fusion

The first implementation scope SHALL not require fused kernels.

Runtime may execute unfused operator sequences.

Fusion may be added later only when graph semantics are preserved.

## No Silent Quantization

Quantized operators SHALL remain placeholders unless explicitly implemented.

Quantized model loading may be blocked, dequantized explicitly, or rejected
according to policy.

No hidden dequantization SHALL occur.

## DType Scope

Initial compute dtype SHOULD prioritize `f32`.

Other dtypes MAY be represented as storage, placeholder, or explicit conversion
targets.

Initial dtype policy SHOULD be:

```text
f32 compute: required
f32 storage: required
i32/u32 token IDs: required for embedding indices
bool masks: required where masks exist
f16/bf16 compute: placeholder unless implemented
int8/uint8 quantized compute: placeholder unless implemented
```

No dtype conversion SHALL happen silently.

## Layout Scope

Initial layout support SHOULD prioritize contiguous layout.

Strided layout MAY be placeholder.

Blocked, packed, paged, and Provider-owned opaque layouts MAY be unsupported.

Initial layout policy SHOULD be:

```text
contiguous: required
strided: placeholder
paged: placeholder
blocked: future
packed quantized: future
provider-owned opaque: Provider internal only
```

## Shape Scope

Initial shape support SHOULD focus on small static or Runtime-known dynamic
shapes.

The first scope SHOULD validate:

- tensor rank
- batch dimension
- sequence length
- hidden size
- head count
- KV head count
- head dimension
- intermediate size
- vocabulary size
- broadcasting policy
- matmul compatibility

Unsupported shape patterns SHALL fail explicitly.

## Attention Scope

Initial attention SHALL be simple reference attention.

It SHOULD support:

- causal attention
- simple mask
- Q/K/V inputs
- f32 accumulation
- softmax
- value aggregation
- optional non-paged KV cache metadata

It MAY reject:

```text
paged KV cache
flash attention
sliding window attention
block sparse attention
multi-query variants if not implemented
grouped-query variants if not implemented
```

If GQA/MQA metadata is not implemented, it SHALL fail explicitly.

## RoPE Scope

Initial RoPE SHALL support one explicit baseline mode.

It SHOULD validate:

- base
- scale
- dimension
- position indices
- tensor shape
- dtype

Unsupported RoPE variants SHALL fail explicitly.

## MLP Scope

Initial MLP path SHOULD be expressible with:

```text
matmul
silu
mul
matmul
```

This covers common gated MLP structures without requiring fused MLP kernels.

GELU may remain placeholder unless required by first target model.

## Residual And Elementwise Scope

Initial elementwise scope SHOULD include:

```text
add
mul
residual-add
```

Broadcasting rules SHALL be explicit.

Unsupported broadcasting SHALL fail.

## Logits Projection Scope

Initial logits projection may use `matmul`.

Sampling remains separate.

The first operator scope SHALL not require Provider-assisted sampling.

## Graph Planning Scope

Execution Graph planning SHALL be able to plan required-now operators.

It SHALL reject or mark unsupported placeholder operators.

Planning SHALL not produce hidden substitutions.

If fallback or conversion is used, it SHALL be explicit.

## Model Component Scope

The first Model Component baseline SHALL only require operators in this scope.

If a Model Component needs an operator outside the scope, Runtime SHALL reject
the model path or report missing operator support.

## Conformance Scope

Each required-now operator SHALL have conformance fixtures.

Fixtures SHOULD be small and deterministic.

Conformance SHOULD validate:

- known output
- invalid shape
- invalid dtype
- invalid layout
- unsupported attribute
- Memory Manager integration
- Kernel Registry discovery
- Reference CPU dispatch

## Error Model

Operator scope errors SHALL be structured.

Error categories SHOULD include:

- operator out of first scope
- operator placeholder only
- operator explicitly unsupported
- first scope dtype unsupported
- first scope layout unsupported
- first scope shape unsupported
- first scope attribute unsupported
- first scope kernel missing
- first scope conformance missing
- first scope conformance failed
- first scope graph planning failed
- internal first operator scope error

## Observability

Runtime SHOULD emit observations for:

- first scope operator accepted
- first scope operator rejected
- placeholder operator encountered
- unsupported operator encountered
- required kernel missing
- dtype unsupported in first scope
- layout unsupported in first scope
- shape unsupported in first scope
- first scope conformance passed
- first scope conformance failed

Observability SHALL not expose raw tensor values, prompts, model weights, KV
cache contents, handles, or memory pointers by default.

## Browser Target

The first operator scope SHALL be platform-neutral.

Browser support is not required.

Browser targets may report unsupported for Reference CPU native paths.

## Non-Goals

This change does not:

- implement all operators
- implement optimized kernels
- implement CUDA kernels
- implement Metal kernels
- implement OpenVINO kernels
- implement QNN kernels
- implement WebGPU kernels
- implement full quantized inference
- implement paged attention
- implement flash attention
- implement speculative decoding
- implement beam search
- implement training or gradients
- implement full Qwen/Llama/Gemma support
- define CLI behavior

## Impact

Magnetar gains a clear first executable operator scope.

The first CPU inference path becomes constrained and testable:

```text
embedding
 → rmsnorm
 → matmul / rope / attention / softmax
 → residual-add
 → rmsnorm
 → matmul / silu / mul / matmul
 → residual-add
 → logits matmul
```

This prepares:

- Qwen model component baseline
- Runtime inference API
- end-to-end local inference conformance