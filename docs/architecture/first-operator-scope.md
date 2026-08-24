# First Operator Implementation Scope

The first operator implementation scope defines the minimum portable Operator
surface Magnetar expects to execute through the correctness-oriented Reference
CPU path. It is Runtime-owned metadata, not Provider-owned metadata.

The scope is platform-neutral. Providers advertise Kernels that may satisfy the
scope, but Provider-specific kernel names, device handles, native memory
pointers, and optimized backend features do not define first-scope semantics.

## Tiers

Operator scope metadata is classified into four execution tiers plus a separate
future-optimized list.

- `required-now`: semantics, validation, Reference CPU coverage, and
  conformance fixtures are required.
- `required-for-first-decoder-model`: the minimal decoder-only model path uses
  these operators. They are a subset of `required-now` except for implemented
  optional operators such as `gelu`, which may exist without being required.
- `placeholder`: the operator identity may exist, but Runtime must not assume
  a Kernel exists. Planning must reject it or carry an explicit pending path.
- `explicitly-unsupported`: Runtime rejects the operator with a structured
  first-scope error.
- `future-optimized`: optimized forms may be added later only if portable
  semantics remain equivalent or variant compatibility is explicit.

## Required-Now Operators

The required-now set is:

- `embedding`
- `matmul`
- `rmsnorm`
- `rope`
- `attention`
- `softmax`
- `silu`
- `add`
- `mul`
- `residual-add`
- `dtype-conversion`
- `layout-conversion`

These operators have Reference CPU kernel coverage and are included in the
first-scope conformance fixture set.

## First Decoder Path

The first decoder-only baseline uses:

```text
embedding
-> rmsnorm
-> matmul / rope / attention / softmax
-> residual-add
-> rmsnorm
-> matmul / silu / mul / matmul
-> residual-add
-> logits matmul
```

Fused MLP, fused RMSNorm, flash attention, tensorcore matmul, and other
optimized operators are not required. Model Components may emit unfused graphs
using the required-now operators.

## Placeholder Operators

The first scope reserves placeholder identities for operators such as
`batched-matmul`, `layernorm`, `dequantize`, `quantize`, `requantize`,
`quantized-matmul`, `paged-attention`, `sampling-helper`,
`logits-processor-helper`, `layout-pack`, and `layout-unpack`.

Placeholders do not imply Kernel availability. Reference CPU required coverage
validation rejects placeholder advertisements unless the operator has been
promoted out of placeholder status with explicit support and conformance.

## Explicitly Unsupported Operators

The first scope rejects operators such as `flash-attention`,
`grouped-quantization`, `moe-dispatch`, speculative decoding helpers, beam
search helpers, training operators, and gradient operators.

Unsupported behavior is explicit. Runtime does not silently emulate unsupported
operators through unrelated operators.

## DType Scope

The first compute dtype baseline is `f32`.

Required first-scope dtypes are:

- `f32` compute and storage
- `i32` and `u32` token IDs where embedding indices require them
- `bool` masks where masks are represented

`f16`, `bf16`, `int8`, and `uint8` compute are placeholders unless explicitly
implemented. Other dtypes are outside the first scope. No dtype conversion is
silent; conversion must be explicit in planning or the graph is rejected.

## Layout Scope

The required layout baseline is contiguous layout.

Strided, paged, and attention-specific layouts are placeholders. Blocked and
packed quantized layouts are future targets. Provider-owned opaque layouts are
Provider-internal only and are not accepted as Component-visible layout
requirements.

No layout conversion is silent; conversion must be explicit in planning or the
graph is rejected.

## Shape Scope

First-scope validation checks tensor rank, batch, sequence length, hidden size,
head count, KV head count, head dimension, intermediate size, vocabulary size,
broadcasting policy, and matmul compatibility where those dimensions apply.
Unsupported shape patterns fail with structured first-scope shape errors.

## Attention Scope

The first attention baseline is simple reference attention with Q/K/V inputs,
f32 accumulation, softmax, value aggregation, causal or bidirectional masking,
and non-paged metadata where implemented.

Paged KV, flash attention, unsupported sliding-window behavior, block sparse
attention, and unsupported GQA/MQA metadata are rejected explicitly.

## RoPE Scope

The first RoPE baseline uses one explicit mode. Base, scale, dimension,
position behavior, shape, and dtype are validated. Unsupported RoPE variants
fail with `first-scope-attribute-unsupported`.

## Error And Observability

First-scope failures use structured error categories such as
`operator-placeholder-only`, `operator-explicitly-unsupported`,
`first-scope-dtype-unsupported`, `first-scope-layout-unsupported`,
`first-scope-shape-unsupported`, `first-scope-kernel-missing`, and
`first-scope-graph-planning-failed`.

Observations identify accepted and rejected operators, placeholder encounters,
unsupported operators, missing kernels, unsupported dtype/layout/shape, and
conformance pass/fail status. Observations carry redacted metadata only; they do
not expose raw tensor values, prompts, weights, KV cache contents, handles,
memory pointers, or Provider-owned resource internals.

## Non-Goals

The first operator scope does not implement all operators, optimized kernels,
CUDA, Metal, OpenVINO, QNN, WebGPU, full quantized inference, paged attention,
flash attention, speculative decoding, beam search, training, gradients, full
Qwen/Llama/Gemma support, or CLI behavior.
