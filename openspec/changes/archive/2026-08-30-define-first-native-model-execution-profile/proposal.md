# Define First Native Model Execution Profile

## Why

Magnetar now has architectural contracts covering substantially more capability
than is required to execute a first model.

The architecture includes:

- portable Model Components
- Model Artifacts
- Tensor Resources
- Runtime Memory Manager
- Operators
- Kernels
- Kernel Registry and Dispatch
- Provider ABI
- Reference CPU Provider
- Provider compilation
- generated Kernel qualification
- Kernel Artifact ingestion
- Kernel specialization
- Runtime autotuning
- adaptive performance feedback
- Prepared Execution Plans
- asynchronous ExecutionStreams
- Device residency
- zero-copy Resources
- Device memory pools
- Allocation Plans
- local multi-Device placement

Continuing to expand every advanced capability before implementing a complete
vertical slice would delay the point at which Magnetar proves its core
architecture through actual model execution.

Magnetar therefore needs a deliberately constrained first implementation
profile.

The profile SHALL define:

```text
what MUST work first
what MAY be implemented first
what is explicitly DEFERRED
```

The purpose is to freeze enough architecture to implement a real Qwen-compatible
Model Component and native Magnetar Kernels without requiring advanced
multi-Device or generated-Kernel infrastructure.

## What Changes

This change defines the First Native Model Execution Profile.

The mandatory first profile uses:

```text
Host:
    single local Runtime

Component Engine:
    native Wasmtime-backed Component Engine

Model architecture:
    Qwen baseline

Model Component:
    WASM Component

Model data:
    external Model Artifact

Provider:
    Reference CPU Provider

Device:
    one logical CPU Device

dtype:
    f32 baseline

execution:
    single Provider
    single Device

Kernel source:
    built-in known-good Magnetar Reference CPU Kernels

generation:
    real tokenization
    real model forward
    incremental decode
    real KV cache
    sampling
    streaming token output

memory:
    TensorResource
    Runtime Memory Manager
    simple pool-backed allocation
    safe workspace reuse

planning:
    ExecutionGraph
    Kernel Registry
    PreparedExecutionPlan

synchronization:
    synchronous Reference CPU implementation
    CompletionToken contract preserved
```

## Definition Of Native

Within this profile, "native Magnetar execution" means that model computation is
executed by Magnetar's own Operator, Kernel, Provider, Tensor, Memory, Planning,
and Generation contracts.

The mandatory conformance path SHALL NOT use Candle to perform model forward
execution.

A temporary Candle Provider MAY continue to exist elsewhere in the repository
for migration purposes.

It SHALL NOT satisfy this profile.

The profile path is:

```text
Model Component
    ->
Execution Graph
    ->
Kernel Registry
    ->
Reference CPU Provider
    ->
Magnetar Reference CPU Kernels
```

and not:

```text
Model Component
    ->
Candle
```

## Stabilization Objective

This profile exists to reach:

```text
ARCHITECTURE FREEZE #1
```

for the local single-Device model-execution path.

Advanced contracts remain architecturally valid but SHALL NOT block this first
implementation milestone.

## Required Execution Path

The mandatory end-to-end path SHALL be:

```text
magnetar-cli / conformance client
        |
        v
RuntimeInferenceApi
        |
        v
Model loading
        |
        v
Qwen Model Component WASM
        |
        v
Execution Graph
        |
        v
Prepared Execution Plan
        |
        v
Kernel Registry
        |
        v
Reference CPU Provider
        |
        v
Reference CPU Kernels
        |
        v
Tensor Resources / Memory Manager
        |
        v
KV Cache
        |
        v
incremental generation
        |
        v
sampling
        |
        v
token stream
```

Every mandatory end-to-end conformance test SHALL exercise this path.

## Runtime Owns Model Execution

RuntimeInferenceApi SHALL own the complete inference execution path.

An inference caller SHALL NOT provide a function equivalent to:

```text
next_logits(tokens) -> logits
```

for ordinary model execution.

Logits SHALL originate from the loaded Model Instance executing its
PreparedExecutionPlan through Kernel dispatch.

## No Caller Supplied Forward Pass

The first profile SHALL remove or isolate any production-facing design in which
the caller computes model logits and Runtime merely performs token-generation
bookkeeping.

Such hooks MAY exist only as explicit test utilities outside the mandatory
inference execution path.

## Qwen Baseline

The first model architecture SHALL use the already-defined Qwen Model Component
contract.

The first profile requires a deliberately small Qwen-compatible fixture rather
than a production-scale model.

The fixture exists to prove architecture and correctness.

It is not a benchmark model.

## Tiny Qwen Fixture

The conformance fixture SHOULD use approximately the following baseline:

```text
architecture:
    decoder-only Qwen-compatible Transformer

vocabulary:
    256 tokens

hidden size:
    64

layers:
    2

attention heads:
    4

KV heads:
    2

head dimension:
    16

intermediate size:
    128

maximum fixture context:
    128 tokens

dtype:
    f32
```

Equivalent small dimensions MAY be used if required by the existing Qwen Model
Component specification, but the fixture dimensions SHALL be:

- deterministic
- small enough for CI
- non-trivial
- capable of exercising GQA/KV behavior
- fixed by conformance data

Changing fixture dimensions after stabilization SHALL update the fixture
version/golden evidence.

## Real Architecture, Small Data

The fixture SHALL execute the real architectural operations required by its Qwen
baseline.

It SHALL NOT replace the model with a fake function such as:

```text
token -> token + 1
```

or:

```text
hard-coded logits
```

The fixture SHOULD include actual:

- embeddings
- RMS normalization
- Q/K/V projections
- RoPE
- attention
- output projection
- gated MLP
- residual paths
- final normalization
- LM-head projection

according to the existing Qwen baseline semantics.

## Deterministic Fixture Weights

Fixture weights SHALL be deterministic.

Weights MAY be:

- generated once from a documented fixed seed and checked into test artifacts
- generated deterministically during fixture creation and persisted
- provided as a fixed test Model Artifact

Runtime execution SHALL consume the Model Artifact.

Kernel tests SHALL NOT obtain special knowledge of the expected weight
generation algorithm.

## Model Artifact

The first profile SHALL use external model data.

The WASM Model Component SHALL NOT embed the complete model weights.

At minimum the fixture SHALL provide:

```text
model configuration
weights
tokenizer artifact/configuration
required model metadata
```

## First Model Artifact Format

The profile MAY use a deliberately constrained physical fixture format.

It SHOULD prefer:

```text
config.json
model.safetensors
tokenizer fixture/artifact
```

where practical.

The first profile does not require:

- sharded safetensors
- remote Hugging Face resolution
- GGUF
- quantized weights
- arbitrary community tokenizer formats
- every Qwen configuration option

The Model Loading layer SHALL still normalize physical model data into the
existing Model Artifact contract.

## Model Component Is WASM

The Qwen architecture implementation SHALL execute as a WASM Component through
the Platform Component Engine abstraction.

For the native first profile, the required engine is the Wasmtime-backed
Component Engine.

Wasmtime types SHALL remain outside public Magnetar Runtime and WIT contracts.

## Model Component Responsibility

The Qwen WASM Model Component owns portable architecture semantics.

It MAY:

- interpret normalized Qwen configuration
- construct architecture-specific graph structure
- describe model tensor relationships
- request portable Operators
- describe prefill/decode computation

It SHALL NOT:

- execute CUDA
- execute AVX directly
- select Provider
- select Device
- allocate native memory
- access native Kernel handles
- receive Device pointers
- choose Reference CPU Kernel IDs
- read arbitrary filesystem paths
- access arbitrary network resources

## Model Component Weight Access

Model Component SHALL interact with Model Artifact weights through Runtime-owned
logical model/tensor Resource contracts.

It SHALL NOT receive arbitrary native memory addresses.

## No WASI Ambient Authority

The Qwen Model Component SHALL not require ambient filesystem/network/process
authority for normal inference.

Required Model Artifact data SHALL be supplied through explicit Runtime
capability/resource contracts.

## Required Operators

The first profile SHALL implement enough Operators to execute the Qwen fixture.

The required baseline catalog is:

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

Where the existing Operator catalog models a Qwen operation through a more
specific already-defined Operator, that canonical Operator SHALL be used.

The first profile SHALL NOT invent duplicate Operator semantics merely to
simplify implementation.

## Operator Correctness First

The first implementation objective is:

```text
correctness
    >
performance
```

Reference Operators/Kernels SHALL favor understandable deterministic
implementations.

Advanced fusion is not required.

## Required Reference CPU Kernels

Reference CPU Provider SHALL provide known-good Kernels sufficient for all
required Operators.

The mandatory baseline uses:

```text
host f32
```

where compatible with the existing Reference CPU Provider specification.

## Kernel Implementation Strategy

The first kernels MAY use straightforward scalar/portable Rust implementations.

SIMD optimization is not required for profile conformance.

The implementation SHALL nevertheless use the normal Kernel contract.

A Reference CPU Kernel SHALL NOT be invoked through special architecture-specific
bypass from Qwen code.

## Kernel Registry Is Mandatory

Qwen execution SHALL resolve Kernels through Kernel Registry/Dispatch.

The E2E path SHALL NOT contain:

```text
reference_cpu::matmul(...)
reference_cpu::attention(...)
```

calls directly from:

- Qwen Model Component
- RuntimeInferenceApi orchestration
- E2E conformance test
- CLI

The Provider implementation itself MAY naturally call its private Kernel
implementation after dispatch has selected it.

## Kernel Selection Baseline

The first profile MAY use a trivial deterministic selection policy because only
one conformant Reference CPU Kernel may exist per Operator.

The selection path SHALL still use the normal Registry/eligibility contract.

This preserves the architecture required for future optimized Kernels.

## Prepared Kernel

Required Reference CPU Kernels SHALL participate in the PreparedKernel
lifecycle.

For simple CPU Kernels, preparation MAY be lightweight.

PreparedKernelId SHALL remain opaque.

## Execution Graph

The Qwen Model Component SHALL result in an actual ExecutionGraph or equivalent
portable Operator graph following the existing execution-graph contract.

The profile SHALL NOT permit a "Qwen forward" monolithic special-case that
bypasses Operator graph semantics.

## Prepared Execution Plan

Runtime SHALL construct a PreparedExecutionPlan for the Qwen execution scope.

The first implementation MAY use a simple Plan:

```text
single Provider
single Device
one compute execution lane
simple resource bindings
```

It SHALL still materialize:

- graph identity
- Kernel bindings
- resource slots
- required execution ordering
- relevant memory requirements

## Plan Reuse

Decode SHOULD reuse a compatible PreparedExecutionPlan rather than repeating
full Kernel discovery for every generated token.

## ExecutionStream Baseline

Reference CPU Provider MAY implement ExecutionStream synchronously.

For the mandatory profile:

```text
submit
    ->
execute immediately
    ->
return completed CompletionToken
```

is valid.

This allows the synchronization contract to be exercised without requiring an
asynchronous task executor in the first implementation.

## CompletionToken Baseline

Even synchronous execution SHALL expose the logical CompletionToken semantics
required by the Runtime.

The profile SHALL therefore test:

- submission
- completion
- dependency propagation
- ResourceReadiness

without requiring parallel CPU scheduling.

## Single Device

The mandatory profile SHALL use exactly one logical Reference CPU Device.

Multi-Device placement contracts remain valid but are not required.

## Memory Manager

All Tensor Resources SHALL be owned through the Runtime Memory Manager.

The first profile SHALL NOT require one sophisticated allocator.

A simple pool/arena-backed implementation is sufficient if it preserves:

- allocation bounds
- alignment
- lifetime
- Resource identity
- alias safety
- ResourceReadiness
- safe reuse

## Memory Pool Baseline

A single host/CPU compatible logical pool MAY satisfy the first profile.

The implementation MAY logically classify allocations into:

- persistent
- KV
- workspace/transient

without requiring independent physical native pools.

## No Per Operator Native Allocation Requirement

The profile SHOULD permit Prepared Plan and Memory Manager reuse so normal
decode does not require an unnecessary operating-system/native allocation for
every Tensor of every Operator.

This is a profile goal rather than a performance benchmark.

## Device Residency Baseline

For Reference CPU, host memory is naturally compatible with the CPU Device.

The first profile SHALL exercise ResourceResidency contracts without requiring
GPU-specific zero-copy features.

## Tensor Views

The first profile SHOULD implement the minimal View semantics required by Qwen
execution.

Views SHALL retain existing:

- bounds checks
- layout
- alias/lifetime semantics

## Dtype

The first profile requires f32 execution.

The architecture SHALL remain dtype-generic.

The first profile does not require:

- f16
- bf16
- fp8
- int8
- int4

for Model execution.

Dtype-conversion Operator SHOULD exist sufficiently to exercise the contract,
but production mixed-precision coverage is deferred.

## Layout

The first profile SHOULD standardize one straightforward dense contiguous
layout for most fixture tensors.

Explicit layout conversion SHALL remain available where the graph requires it.

Advanced tiled Provider-specific layouts are deferred.

## Tokenizer

The Runtime SHALL use the existing Tokenizer contract.

The first fixture SHALL have deterministic tokenization and detokenization.

A trivial byte/fixture tokenizer MAY be used if represented through the normal
Tokenizer contract.

It SHALL NOT bypass tokenization by injecting final token IDs directly into the
mandatory CLI E2E test.

Focused lower-level tests MAY naturally operate on token IDs.

## Prompt Path

The mandatory CLI/conformance path SHALL begin from text.

Example:

```text
"Hello"
    ->
Tokenizer
    ->
token IDs
    ->
model execution
```

## Generation

The first profile SHALL execute the normal Generation contract.

Generation SHALL include:

- prompt tokenization
- prefill
- incremental decode
- logits production
- sampling
- token append
- stop condition
- output decoding/streaming

## Prefill

Initial prompt tokens SHALL execute through the real model path.

Prefill SHALL establish KV-cache state used by decode.

## Incremental Decode

Decode SHALL process newly generated tokens using existing KV state.

It SHALL NOT recompute the entire prompt and generated sequence for every token
as the mandatory execution strategy.

The correctness invariant is:

```text
prefill prompt
      |
      v
KV state
      |
      v
decode token N
      |
      +-- read prior KV
      +-- append new KV
      |
      v
decode token N+1
```

## Real KV Cache

The profile SHALL use the existing Runtime-owned KV Cache contract.

KV state SHALL not be hidden inside:

- CLI
- test closure
- Qwen-specific global state
- caller callback

The Runtime/Session execution path owns KV lifecycle.

## KV Layout

The first profile MAY use a simple contiguous KV layout.

Paged KV cache is not mandatory.

The representation SHALL still maintain:

- Session ownership
- layer identity
- sequence position
- K/V distinction
- shape/dtype correctness
- append/read correctness

## Attention

The required Attention implementation SHALL consume the actual prior KV state
during decode.

A fake Attention implementation that ignores KV history SHALL not satisfy the
profile.

## RoPE

RoPE position during incremental decode SHALL use the actual token position.

The profile SHALL include regression tests for non-zero decode positions.

## Sampling

The first profile SHALL support deterministic sampling suitable for
conformance.

At minimum it SHALL support greedy sampling.

Existing more advanced sampling contracts MAY remain implemented.

Greedy mode SHOULD be the canonical deterministic E2E conformance mode.

## RNG

Stochastic sampling is not required for the first profile's primary E2E test.

If stochastic sampling is exposed, it SHALL obey the existing explicit Runtime
RNG-state contract.

## Session

The mandatory inference path SHALL use the existing Inference Session model.

Session SHALL own or reference:

- generation state
- KV state
- cancellation state
- relevant execution scope

It SHALL not become a CLI agent/tool Session.

## RuntimeInferenceApi

RuntimeInferenceApi SHALL provide sufficient operations for:

```text
load model
create inference/session context
generate
stream output
cancel
close/release
```

The exact API remains governed by the existing Runtime Inference API contract.

## CLI

`magnetar-cli` SHALL remain a client of RuntimeInferenceApi.

It SHALL NOT:

- execute Kernels
- call Provider directly
- construct CPU logits
- own model KV
- load Reference CPU Kernel functions directly

## Mandatory CLI Command

A first-profile implementation SHOULD support a command conceptually equivalent
to:

```text
magnetar run <fixture-model> "Hello"
```

The physical fixture-model reference MAY evolve.

The important invariant is that CLI reaches the model only through
RuntimeInferenceApi.

## Mandatory E2E Conformance

The first profile SHALL provide one deterministic end-to-end conformance test.

The test SHALL start from text and finish with generated text/token output.

It SHALL traverse:

```text
client
RuntimeInferenceApi
Tokenizer
Model Instance
Qwen WASM Component
Execution Graph
Prepared Execution Plan
Kernel Registry
Reference CPU Provider
Reference CPU Kernels
Memory Manager
KV Cache
Generation
Sampling
```

## Structural E2E Evidence

The E2E suite SHALL not infer architectural compliance solely from final output.

It SHALL collect safe structured evidence proving that expected layers were
traversed.

Evidence MAY include:

- Model Component loaded
- graph created
- Plan prepared
- Registry resolution occurred
- Reference CPU Provider selected
- required Kernels executed
- KV prefill occurred
- incremental KV append occurred
- decode reused prior KV
- Generation emitted token

## No Direct Kernel E2E Bypass

The E2E test SHALL fail profile conformance if it computes expected model output
by directly calling Reference CPU Kernel functions in place of Runtime
execution.

Golden/reference calculations MAY exist independently for comparison.

They SHALL not constitute the System Under Test path.

## Golden Results

The fixture SHOULD provide deterministic golden evidence.

Golden evidence MAY include:

- tokenization output
- selected intermediate Operator outputs
- prefill logits
- decode logits
- greedy token sequence
- final decoded output

Golden values SHALL be versioned with fixture/model semantics.

## Differential Reference

Focused Kernel/Operator tests MAY compare implementation output against simple
independent mathematical reference implementations.

The System Under Test SHALL still traverse normal Kernel contract where
conformance requires it.

## Required Failure Cases

The first profile SHOULD exercise at least:

- invalid model configuration
- missing weight
- incompatible Tensor shape
- unavailable Kernel
- invalid Prepared Plan
- KV position error
- cancellation
- invalid token ID
- malformed Component Artifact

Failures SHALL be structured.

## Required Security Boundary

The Qwen WASM Component SHALL execute without ambient network or filesystem
authority.

The component SHALL not be able to obtain:

- Runtime secrets
- arbitrary process environment
- native Provider handles
- native Tensor pointers

## Required Observability

The profile SHOULD emit enough safe observations to prove the E2E path without
logging:

- model weights
- Tensor contents by default
- prompts unless explicitly permitted
- KV contents
- native pointers
- secrets

## Deferred Features

The following contracts remain valid but SHALL NOT be mandatory for First
Native Model Execution Profile conformance:

```text
multi-Device placement
Tensor Parallel
collective execution
multi-host execution

generated Kernels
Provider runtime compilation
Kernel Artifact ingestion
production Kernel hot swap
Kernel canary promotion

Runtime autotuning
adaptive re-tuning
Performance Model-driven replacement

CUDA Provider
Metal Provider
OpenVINO Provider
QNN Provider
WebGPU Provider

f16
bf16
fp8
quantization

cross-Provider zero-copy
peer Device access
peer Device transfer

memory compaction
memory overcommit
advanced pool borrowing

paged KV cache
Prefix Cache optimization
continuous batching optimization

Provider-native graph capture
advanced asynchronous ExecutionStreams
```

Implementations MAY support any deferred feature.

Missing deferred features SHALL NOT fail this profile.

## Advanced Contracts Must Not Block Baseline

A Runtime implementation SHALL be allowed to implement simplified conformant
forms of generic contracts for the baseline.

Examples:

```text
ExecutionStream
    -> synchronous CPU stream

DeviceMemoryPool
    -> simple host-backed pool

AllocationPlan
    -> conservative slots

Kernel Selection
    -> one eligible Kernel

ResourceResidency
    -> host/CPU resident
```

The simplification SHALL preserve the semantic contract.

## No Fake Abstractions

Simplification SHALL NOT mean bypass.

For example:

Valid:

```text
Kernel Registry
    -> exactly one Reference CPU candidate
```

Invalid:

```text
Qwen code
    -> directly calls cpu_matmul()
```

Valid:

```text
ExecutionStream
    -> synchronous completed token
```

Invalid:

```text
no execution/completion contract exists at all
```

## Stabilization Boundary

Once this profile and its corresponding implementation/conformance cut are
accepted, new advanced architectural changes SHOULD NOT block implementation of
the first native Qwen vertical slice unless they identify:

- correctness flaw
- security flaw
- impossible implementation boundary
- ABI-breaking foundational issue

Feature expansion SHOULD proceed after the first vertical slice.

## Compatibility With Future Providers

The baseline SHALL avoid CPU-specific behavior in public Runtime contracts.

Reference CPU is the first realization, not the architectural definition of
execution.

A future accelerated Provider SHALL be able to reuse:

- Model Component
- Execution Graph
- RuntimeInferenceApi
- Model Artifact
- Session
- Generation
- Kernel Registry
- Prepared Execution Plan

without changing Qwen semantic implementation.

## Success Criterion

The profile is satisfied when a deterministic Qwen-compatible fixture can
execute from text prompt to generated output through the real Magnetar
architecture without:

- Candle model execution
- caller-supplied logits
- direct Kernel E2E bypass
- fake KV
- full-sequence decode recomputation as the required path

Conceptually:

```text
magnetar run qwen-fixture "Hello"
```

shall demonstrate:

```text
Magnetar owns AI execution.
```

## Non-Goals

This change does not:

- implement the profile
- define the exact PR sequence
- define release packaging
- optimize Reference CPU performance
- define Tensor Parallel
- define collectives
- define accelerated Provider requirements
- require production-scale Qwen
- require Hugging Face Hub
- require quantization
- require continuous batching
- require generated Kernels
- reopen previously established architectural boundaries

## Impact

Magnetar moves from:

```text
broad architecture specification
```

to:

```text
bounded implementation target
```

The first implementation can now focus on a real vertical slice instead of
waiting for every advanced execution capability.