# First Native Model Execution Profile

The First Native Model Execution Profile is the bounded baseline for the first
complete local model execution path.

Profile version: `0.1.0`

The profile requires a single local Runtime, the Platform Component Engine, a
native Wasmtime-backed component profile, a Qwen WebAssembly Model Component,
external model artifact data, Runtime-owned tokenization/generation/session
state, the Kernel Registry, Reference CPU Provider, one logical CPU Device,
f32 Tensor execution, Runtime Memory Manager ownership, KV cache, incremental
decode, greedy sampling, streaming output, and redacted structural evidence.

## Native Execution

Native Magnetar execution means model computation flows through Magnetar
Operator, Kernel, Provider, Tensor, Memory, Planning, Session, Generation, and
Sampling contracts.

The conformant path is:

```text
RuntimeInferenceApi
  -> Model Loading
  -> Qwen WebAssembly Model Component
  -> Execution Graph
  -> Prepared Execution Plan
  -> Kernel Registry
  -> Reference CPU Provider
  -> Reference CPU Kernels
  -> Tensor Resources / Memory Manager
  -> KV Cache
  -> incremental generation
  -> greedy sampling
  -> token stream
```

Candle may remain in the repository for migration, but it does not satisfy this
profile and must not appear in the mandatory model-forward evidence path.

## Qwen Fixture

The first fixture is a small deterministic Qwen-compatible decoder-only model.
It is intended for CI correctness, not benchmark performance.

Current fixture version: `0.1.0`

Current fixture dimensions:

```text
vocabulary: 258 tokens including fixture special tokens
hidden size: 4
layers: 1
attention heads: 2
KV heads: 2
head dimension: 2
intermediate size: 8
maximum context: 32 tokens
dtype: f32
```

Fixture weights are deterministic and generated from stable tensor names and
indices. The versioned weight digest is:

```text
sha256:ed7d3a310ae30e08f170ed61cd73f9053e498ae9a17dd7dc980fd61a3152ed90
```

The fixture uses a deterministic byte tokenizer through the normal Tokenizer
contract. Mandatory CLI and E2E paths begin from text; focused lower-level tests
may operate directly on token IDs.

## Required Operators And Kernels

The mandatory Operator set is:

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

Reference CPU provides correctness-first f32 kernels for this set. Kernels are
advertised and selected through the normal Kernel Registry and Dispatch path.
Qwen code, CLI code, RuntimeInferenceApi callers, and E2E tests must not call
Reference CPU kernel functions as the System Under Test path.

## Incremental KV

Prefill executes prompt tokens through the real model path and populates
Runtime-owned KV cache state. Decode consumes exactly the new token plus prior
KV state, appends new K/V entries, and advances the actual sequence position.
The mandatory profile does not accept full-sequence recomputation as the
required decode strategy.

## Deferred Capabilities

The following capabilities remain valid architecture but are not required for
this profile:

```text
multi-Device placement
Tensor Parallel
collectives
generated Kernels
Provider runtime compilation
Kernel Artifact ingestion
hot swap
Runtime autotuning
adaptive performance feedback
Performance Model replacement
accelerated Providers
f16, bf16, fp8, int8, int4
quantization
cross-Provider zero-copy
advanced memory pools
paged KV cache
Prefix Cache optimization
production continuous batching optimization
advanced asynchronous ExecutionStreams
```

Missing deferred features must not fail First Native Model Execution Profile
conformance.
