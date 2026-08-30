# Define First Native Model Implementation And Conformance Cut

## Why

The First Native Model Execution Profile defines the bounded architecture that
Magnetar must implement for its first real model vertical slice.

The remaining risk is no longer architectural completeness.

The risk is implementation drift.

Without an explicit implementation cut, development could proceed in an order
that produces apparently working pieces while preserving the exact architectural
problems that the OpenSpec design was intended to eliminate.

Examples include:

- implementing Qwen before Kernel Registry dispatch is real
- letting E2E tests call Reference CPU Kernels directly
- leaving `next_logits` caller callbacks in RuntimeInferenceApi
- implementing generation around full-sequence recomputation
- hiding KV state inside Qwen-specific code
- introducing a monolithic Qwen forward function
- implementing CPU functions without the Provider/PreparedKernel lifecycle
- connecting the CLI to temporary logits rather than Runtime-owned execution
- declaring success from final text output without structural conformance
  evidence

The project therefore needs one final stabilization change that defines:

```text
implementation order
dependency gates
integration gates
conformance gates
cutover criteria
```

This change does not add a new execution abstraction.

It converts the First Native Model Execution Profile into an executable
implementation program.

## What Changes

This change defines:

- implementation phases
- mandatory pull-request boundaries
- dependency ordering
- temporary compatibility rules
- architectural bypass prohibitions
- integration gates
- conformance gates
- deterministic fixture requirements
- golden evidence requirements
- test layering
- failure-path requirements
- CI requirements
- implementation evidence
- stabilization exit criteria
- Definition of Done
- architecture-freeze rules after acceptance

## Core Objective

The implementation milestone is:

```text
magnetar run qwen-fixture "Hello"
```

executing through:

```text
magnetar-cli
    |
    v
RuntimeInferenceApi
    |
    v
Tokenizer
    |
    v
Model Loading
    |
    v
Qwen WASM Model Component
    |
    v
Execution Graph
    |
    v
Prepared Execution Plan
    |
    v
Kernel Registry / Dispatch
    |
    v
Reference CPU Provider
    |
    v
Prepared Reference CPU Kernels
    |
    v
Tensor Resources / Memory Manager
    |
    v
real KV Cache
    |
    v
incremental decode
    |
    v
Sampling
    |
    v
generated token stream
```

The milestone SHALL NOT be considered complete if any mandatory layer is
replaced by a test-specific shortcut.

## Architecture Freeze #1

Acceptance of this change establishes:

```text
ARCHITECTURE FREEZE #1
```

for the first native local single-Device model-execution path.

After the freeze, new architecture changes SHOULD NOT block implementation
unless they resolve one of:

- correctness blocker
- security blocker
- impossible implementation contract
- unavoidable ABI break
- contradiction between already accepted specifications

Feature expansion SHALL be deferred until the vertical slice is operational.

## Implementation Philosophy

Implementation SHALL proceed bottom-up enough to make architectural boundaries
real before model integration.

The required dependency order is:

```text
foundation
    |
    v
Tensor / Memory
    |
    v
Operators
    |
    v
Reference CPU Kernels
    |
    v
Provider / PreparedKernel
    |
    v
Kernel Registry / Dispatch
    |
    v
Execution Plan / Stream
    |
    v
Model Artifact / Loading
    |
    v
Qwen WASM Component
    |
    v
KV / Incremental Execution
    |
    v
Generation / Sampling
    |
    v
RuntimeInferenceApi
    |
    v
CLI
    |
    v
Full E2E Conformance
```

A later layer MAY be developed in parallel where interfaces are stable, but it
SHALL NOT claim integration completion before its required lower-layer gates are
satisfied.

## Implementation Phases

The mandatory implementation cut SHALL contain the following phases:

```text
Phase 0  Freeze and baseline cleanup
Phase 1  Runtime execution foundations
Phase 2  Tensor and Memory
Phase 3  Operator catalog
Phase 4  Reference CPU Kernels and Provider
Phase 5  Registry, Dispatch, Preparation
Phase 6  Execution Graph and Prepared Execution Plan
Phase 7  Model Artifact and deterministic fixture
Phase 8  Qwen WASM Model Component
Phase 9  KV Cache and incremental model execution
Phase 10 Generation and Sampling
Phase 11 RuntimeInferenceApi ownership
Phase 12 CLI integration
Phase 13 Full E2E conformance
Phase 14 Stabilization and cutover
```

## Phase 0 — Freeze And Baseline Cleanup

Before implementing the vertical slice, the repository SHALL identify and
isolate architectural bypasses that would make the final integration ambiguous.

At minimum, the implementation SHALL identify:

- caller-provided logits callbacks
- CLI placeholder logits
- direct Reference CPU Kernel E2E calls
- Qwen-specific direct CPU execution paths
- full-sequence decode-only paths
- test fixtures that bypass Model Artifact loading
- test fixtures that bypass WASM Component loading

These paths MAY temporarily remain during migration only when:

- clearly marked non-conformant/deprecated/test-only
- excluded from mandatory conformance
- not used by the final vertical slice
- tracked for removal

## Phase 1 — Runtime Execution Foundations

The Runtime SHALL have concrete baseline types and lifecycle for:

- Runtime execution context
- Provider registry
- Device identity
- Tensor Resource identity
- PreparedKernelId
- PreparedExecutionPlan identity
- ExecutionStream
- CompletionToken
- structured execution errors

The first implementation MAY remain synchronous.

The important requirement is that the architectural lifecycle exists.

## Phase 2 — Tensor And Memory

Tensor and Memory SHALL be implemented before model-specific execution.

The implementation SHALL provide:

- TensorDescriptor
- TensorResource
- shape validation
- dtype
- layout
- logical byte-size calculation
- overflow-safe size arithmetic
- Resource lifetime
- simple View support
- Memory Manager allocation
- simple pool/arena
- safe transient reuse
- alignment
- ResourceReadiness

All model tensors SHALL eventually pass through these contracts.

## Phase 3 — Operator Catalog

The required Operator semantics SHALL be implemented independently from Qwen.

Mandatory Operators are:

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

Operator validation SHALL precede unsafe Kernel execution.

## Phase 4 — Reference CPU Kernels And Provider

Reference CPU Kernels SHALL be implemented as correctness-first native Magnetar
Kernels.

This phase SHALL establish:

```text
Operator
    |
    v
Kernel implementation
    |
    v
Reference CPU Provider
```

without Qwen-specific calls.

The Provider SHALL remain model-family neutral.

## Phase 5 — Registry Dispatch And Preparation

Reference CPU Kernels SHALL be registered through Kernel Registry.

Kernel registration errors SHALL not be silently discarded.

The Runtime SHALL perform:

```text
Operator requirement
    |
    v
Kernel Registry
    |
    v
eligibility
    |
    v
selection
    |
    v
PreparedKernel
    |
    v
Provider execution
```

The first selection algorithm MAY be trivial and deterministic.

The path itself is mandatory.

## Phase 6 — Execution Graph And Prepared Execution Plan

Execution Graph SHALL become executable through normal Runtime contracts.

PreparedExecutionPlan SHALL materialize:

- graph identity
- node bindings
- selected Kernel identities
- PreparedKernel identities
- CPU Device binding
- resource slots
- execution order
- simple AllocationPlan
- synchronous ExecutionStream binding

The Plan SHALL be reusable across compatible decode steps.

## Phase 7 — Model Artifact And Fixture

The deterministic Qwen fixture SHALL be finalized before full Model Component
integration.

The fixture SHALL include:

- versioned config
- versioned deterministic weights
- tokenizer data/config
- model metadata
- fixture digest

The test model SHALL be small enough for normal CI.

## Phase 8 — Qwen WASM Model Component

The Qwen architecture SHALL be implemented as a real WASM Component.

The Component SHALL construct/use portable Qwen architecture semantics through
the Operator/graph boundary.

The Component SHALL NOT:

- call Reference CPU implementation functions
- select Provider
- select Device
- perform native allocation
- receive native pointers
- embed special direct model-forward authority into Provider

## Phase 9 — KV Cache And Incremental Model Execution

The implementation SHALL establish real stateful decode.

The required model execution phases are:

```text
PREFILL
    process prompt tokens
    create/populate KV

DECODE
    process new token
    read existing KV
    append new KV
    advance position
```

The mandatory decode path SHALL NOT recompute the complete historical sequence
for every generated token.

## Phase 10 — Generation And Sampling

Generation SHALL consume model-produced logits.

Greedy sampling SHALL be mandatory.

The Generation layer SHALL coordinate:

- prefill
- logits
- sampling
- token append
- incremental decode
- stop condition
- output streaming

Generation SHALL NOT itself implement Qwen forward execution.

## Phase 11 — RuntimeInferenceApi Ownership

RuntimeInferenceApi SHALL become the sole normal high-level execution entry
point for first-profile inference.

Any API where caller provides model-forward logic or `next_logits` SHALL be
removed from the normal path or isolated as a clearly non-production test
utility.

The API SHALL own orchestration of:

```text
Model Instance
Session
Tokenizer
Generation
Execution
KV
```

## Phase 12 — CLI Integration

`magnetar-cli` SHALL become a pure inference client.

The CLI SHALL:

- resolve user model reference/input
- pass prompt/configuration
- call RuntimeInferenceApi
- render streaming/final output

The CLI SHALL NOT:

- create fake logits
- call Kernel
- call Provider
- manage KV
- perform Qwen forward

## Phase 13 — Full E2E Conformance

The final conformance System Under Test SHALL begin from text and terminate in
generated output.

The E2E path SHALL be real.

No intermediate architecture layer may be replaced by a convenience callback.

## Phase 14 — Stabilization And Cutover

After full E2E passes, the implementation SHALL:

- remove or isolate superseded bypasses
- freeze golden fixture
- stabilize public API names used by the profile
- document known deferred features
- run full CI/conformance
- record evidence
- establish implementation baseline commit/tag/reference

This does not imply a public stable release.

It establishes the first implementation baseline.

## Pull Request Sequence

The preferred implementation sequence is:

```text
PR 01 — Freeze profile and remove/mark bypasses
PR 02 — Tensor Resource foundations
PR 03 — Memory Manager and simple pool
PR 04 — Operator catalog and validation
PR 05 — Reference CPU elementwise/norm Kernels
PR 06 — Reference CPU MatMul/Embedding Kernels
PR 07 — RoPE/Softmax/Attention Kernels
PR 08 — Reference CPU Provider + PreparedKernel
PR 09 — Kernel Registry + Dispatch
PR 10 — Execution Graph executor
PR 11 — PreparedExecutionPlan + synchronous ExecutionStream
PR 12 — Deterministic Model Artifact fixture
PR 13 — Qwen WASM Model Component
PR 14 — Model Loading / Model Instance integration
PR 15 — Runtime-owned KV Cache
PR 16 — Real Prefill + Incremental Decode
PR 17 — Sampling + Generation
PR 18 — RuntimeInferenceApi cutover
PR 19 — magnetar-cli cutover
PR 20 — Full E2E conformance and cleanup
```

The exact number of PRs MAY change.

The dependency and conformance boundaries SHALL not.

## PR 01 Gate

PR 01 SHALL establish a clear migration inventory.

It SHOULD identify code implementing:

```text
caller-provided next_logits
placeholder CLI logits
direct E2E CPU Kernel execution
full-sequence decode shortcuts
```

Each SHALL be:

- removed
- deprecated
- isolated
- or explicitly tracked for removal before PR 20

## PR 02 Gate — Tensor

Required tests SHALL prove:

- safe shape arithmetic
- byte-size overflow rejection
- dtype/layout identity
- Tensor Resource identity
- View bounds
- no native pointer in public descriptor

No Qwen integration is required yet.

## PR 03 Gate — Memory

Required tests SHALL prove:

- allocations are Memory Manager-owned
- Resource lifetime is safe
- alignment is respected
- released in-flight Resource is not reused
- simple allocation reuse works
- no Tensor payload is owned by arbitrary Qwen-specific allocator

## PR 04 Gate — Operators

Each mandatory Operator SHALL have:

- semantic descriptor
- input validation
- output shape inference/validation where applicable
- dtype/layout constraints
- negative tests

## PR 05 Gate — Basic CPU Kernels

The first Kernel subset SHOULD include:

```text
add
mul
residual-add
silu
rmsnorm
```

These Kernels SHALL be tested independently against mathematical references.

## PR 06 Gate — MatMul And Embedding

MatMul SHALL receive extensive correctness coverage because most later model
operations depend on it.

Tests SHOULD include:

- rectangular shapes
- small matrices
- zero/invalid dimension behavior according to contract
- shape mismatch
- deterministic f32 results

Embedding SHALL validate token/index bounds.

## PR 07 Gate — RoPE Softmax Attention

These Kernels SHALL include regression cases for:

- non-zero RoPE position
- causal masking
- softmax numerical stability
- grouped-query KV mapping
- incremental KV input

Attention SHALL NOT be considered complete if it ignores historical KV state.

## PR 08 Gate — Provider

Reference CPU Provider SHALL:

- expose one CPU Device
- advertise required capabilities
- prepare Kernel
- execute PreparedKernel
- return structured errors
- support synchronous CompletionToken semantics

## PR 09 Gate — Registry

Tests SHALL prove:

- all required Reference CPU Kernels register
- duplicate/invalid registration failures are visible
- eligibility works
- correct Kernel is selected
- Provider execute path is reached
- direct function call is unnecessary

## PR 10 Gate — Execution Graph

A synthetic graph independent from Qwen SHALL execute through:

```text
Execution Graph
    ->
Registry
    ->
Provider
    ->
Tensor output
```

before Qwen integration.

## PR 11 Gate — Prepared Plan

A synthetic multi-node graph SHALL execute repeatedly through one compatible
PreparedExecutionPlan.

Tests SHALL prove:

- Kernel bindings are reused
- resource slots are reused where safe
- Plan guard failures are structured
- ExecutionStream returns CompletionToken
- no full Registry selection is required for every repeated execution

## PR 12 Gate — Fixture

The fixture SHALL be frozen enough to support deterministic golden testing.

It SHALL include:

```text
FixtureVersion
ModelArtifactDigest
ConfigDigest
WeightsDigest
TokenizerDigest
```

Equivalent stable identity representation is acceptable.

## PR 13 Gate — Qwen Component

Tests SHALL prove:

- WASM Component instantiates
- Qwen config is accepted
- invalid config is rejected
- graph/architecture output is correct
- no Provider/Device concrete identity is embedded
- no ambient filesystem/network access is required

## PR 14 Gate — Model Instance

The loaded model SHALL combine:

```text
Model Artifact
+
Qwen Model Component
+
Provider/Device resolution
+
Prepared Execution Plan
```

and reach READY state.

No generation is required yet.

## PR 15 Gate — KV

Tests SHALL prove:

- per-layer K/V storage
- append
- read
- sequence position
- Session isolation
- context bounds
- prior state survives between model steps

## PR 16 Gate — Prefill And Incremental Decode

This is a critical architecture gate.

The implementation SHALL prove separately:

```text
prefill(prompt)
    ->
KV length = prompt length
```

and:

```text
decode(new_token)
    ->
reads prior KV
    ->
appends exactly one new position
```

or the equivalent number of newly processed positions.

A test SHALL fail if the mandatory decode implementation recomputes the entire
history.

## PR 17 Gate — Generation

Generation SHALL:

- call model execution
- receive logits
- greedy sample
- append generated token
- invoke incremental decode
- stop deterministically

Tests SHALL prove logits originate from model execution rather than fixture
callbacks.

## PR 18 Gate — RuntimeInferenceApi

The normal API SHALL no longer need caller model-forward authority.

A compile-time or runtime architecture test SHOULD prove the ordinary public
generation path has no `next_logits` callback requirement.

## PR 19 Gate — CLI

The CLI command SHALL be capable of:

```text
magnetar run <fixture> "..."
```

or equivalent.

The CLI test SHALL use RuntimeInferenceApi and SHALL not link a fake model
executor.

## PR 20 Gate — Full E2E

The first vertical slice SHALL execute from prompt text through generated token
output.

The E2E SHALL gather safe structural evidence for every mandatory architecture
layer.

## Test Pyramid

The implementation SHALL use several distinct test layers.

### Layer 1 — Mathematical Kernel Unit Tests

Purpose:

```text
does the Kernel compute the right result?
```

These MAY call Kernel implementation at the appropriate Kernel-unit boundary.

They are not E2E architecture tests.

### Layer 2 — Provider Conformance Tests

Purpose:

```text
does Provider correctly prepare and execute Kernels?
```

These SHALL use Provider interfaces.

### Layer 3 — Registry And Dispatch Tests

Purpose:

```text
does Runtime discover/select/dispatch Kernel correctly?
```

These SHALL use Registry/Dispatch.

### Layer 4 — Execution Graph Tests

Purpose:

```text
does a portable Operator graph execute correctly?
```

These SHALL use Runtime graph execution.

### Layer 5 — Model Component Tests

Purpose:

```text
does the Qwen Component describe the correct architecture?
```

These SHALL execute Component boundary.

### Layer 6 — Model Execution Tests

Purpose:

```text
does loaded Qwen fixture produce correct logits/KV?
```

These SHALL use Model Instance and Runtime execution.

### Layer 7 — RuntimeInferenceApi Tests

Purpose:

```text
does Runtime own end-to-end inference orchestration?
```

### Layer 8 — CLI E2E

Purpose:

```text
does the user-facing path traverse all required layers?
```

A lower-layer test SHALL NOT be presented as proof of a higher-layer boundary.

## Independent Mathematical Reference

Reference CPU Kernel outputs SHOULD be compared against independent simple
mathematical implementations or versioned expected values.

The independent oracle SHOULD NOT simply invoke the same production Kernel
function twice.

## Numerical Tolerance

Each numerical Operator SHALL define appropriate f32 tolerance.

Golden tests SHALL avoid unrealistic exact-bit equality where floating-point
operation order legitimately differs.

Deterministic scalar Reference CPU implementations MAY use tighter tolerances.

## Structural Evidence

Final E2E conformance SHALL produce or internally collect an evidence record.

Conceptually:

```text
NativeExecutionEvidence
    component_loaded
    model_artifact_loaded
    graph_fingerprint
    plan_generation
    registry_resolutions
    provider_id
    kernel_ids
    prefill_token_count
    kv_length_after_prefill
    decode_steps
    kv_growth
    generated_token_ids
```

The exact structure is implementation-defined.

Sensitive values SHALL be omitted/redacted according to existing observability
contracts.

## Required E2E Assertions

The final test SHALL assert more than output text.

It SHALL prove at least:

```text
Qwen WASM Component loaded
Model Artifact loaded
Execution Graph built
Prepared Execution Plan ready
Kernel Registry resolved required Kernels
Reference CPU Provider executed work
prefill created KV
decode reused existing KV
decode advanced position
Generation sampled model-produced logits
CLI/API returned generated output
```

## Bypass Detection

The conformance suite SHOULD make bypasses observable.

Where practical, test instrumentation SHALL fail if:

- Reference CPU Kernel is called outside Provider path during E2E
- no Registry resolution exists
- Component load did not occur
- logits provenance is external callback
- KV length is reset/rebuilt incorrectly
- Candle Provider participates in native profile execution

## Temporary Migration Adapters

Temporary adapters MAY exist while PRs are landing.

They SHALL:

- be clearly marked
- not become public architectural contract
- not participate in final conformance
- have removal criteria

## Feature Flags

Advanced features MAY remain feature-gated.

The first-profile CI SHOULD have a minimal feature configuration proving the
vertical slice does not accidentally depend on:

- Candle
- CUDA
- generated Kernel tooling
- autotuning
- multi-Device execution

## Required Build Profile

A canonical minimal build SHOULD exist.

Conceptually:

```text
cargo test --workspace <first-profile-feature-set>
```

Exact Cargo features are implementation-specific.

The profile SHALL be reproducible in CI.

## CI Gates

Before the implementation cut is considered complete, CI SHALL require:

- rustfmt
- cargo check
- clippy
- unit tests
- Provider conformance
- Runtime conformance
- Component tests
- native first-profile E2E
- OpenSpec validation
- WIT validation
- coverage policy already established for project
- dependency/security checks already required by project policy

## No Test-Only Architecture

Production architecture SHALL not rely on `cfg(test)` implementations for
mandatory runtime functionality.

Fixture data may naturally be test-only.

Execution contracts used by E2E SHALL be production code paths.

## Failure Path Coverage

Mandatory structured failure coverage SHALL include at least:

```text
invalid Tensor shape
allocation failure
Kernel unavailable
Provider unavailable
invalid Model Artifact
missing model weight
Component instantiation failure
invalid Qwen configuration
KV overflow
invalid decode position
cancelled generation
```

## Cancellation Gate

The first profile MAY execute synchronously, but cancellation semantics SHALL
still be coherent.

If cancellation cannot interrupt an already executing scalar Kernel, Runtime
MAY stop future generation work after the current synchronous operation.

The API SHALL still return cancelled state according to contract.

## Observability Gate

The vertical slice SHALL provide sufficient observability to diagnose:

- model load
- Component load
- Plan preparation
- Kernel resolution
- Provider execution
- generation progress
- KV progression

without requiring payload logging.

## Performance Is Not Exit Gate

The first implementation cut SHALL NOT require production performance.

The exit priority is:

```text
architecture
correctness
conformance
determinism
failure safety
```

before:

```text
throughput
latency optimization
SIMD
accelerator support
```

## No Premature Kernel Optimization

Reference CPU Kernels SHOULD remain simple until their semantics are covered by
tests.

Optimization that materially complicates correctness review SHOULD be deferred
until after the baseline is green.

## Implementation Done Criteria

A subsystem task is not DONE merely because code exists.

It SHALL satisfy:

```text
implementation
+
unit tests
+
negative tests
+
integration at its declared layer
+
structured errors
+
required observability
```

where applicable.

## Vertical Slice Done Criteria

The first native vertical slice is DONE only if all of the following are true:

```text
[1] Qwen fixture is a real Model Artifact.

[2] Qwen architecture is executed through a WASM Component.

[3] Model execution produces an Execution Graph.

[4] Runtime builds a Prepared Execution Plan.

[5] Required Operators resolve through Kernel Registry.

[6] Reference CPU Provider owns actual Kernel execution.

[7] All required first-profile Kernels are native Magnetar Kernels.

[8] Tensor storage is Memory Manager-owned.

[9] Prefill creates real KV state.

[10] Decode consumes previous KV.

[11] Decode does not require full historical recomputation.

[12] RoPE uses actual incremental position.

[13] Model-produced logits feed Sampling.

[14] Greedy generation produces deterministic golden tokens.

[15] RuntimeInferenceApi requires no caller model-forward callback.

[16] magnetar-cli calls RuntimeInferenceApi.

[17] Native profile does not use Candle model execution.

[18] Final E2E test uses no direct Reference CPU Kernel bypass.

[19] Structural evidence proves the mandatory path.

[20] Minimal CI configuration passes.
```

Failure of any mandatory criterion means the vertical slice is not complete.

## Architecture Freeze Exit

After the above criteria pass:

```text
FIRST NATIVE IMPLEMENTATION BASELINE
```

is established.

At that point the project MAY reopen feature design for:

- optimized CPU
- CUDA
- Metal
- quantization
- generated Kernels
- autotuning
- multi-Device
- local collectives
- continuous batching optimization

These features SHOULD build on the validated vertical architecture rather than
replacing it.

## First Post-Baseline Recommendation

The first post-baseline optimization SHOULD normally be one of:

```text
optimized CPU Provider
```

or:

```text
first accelerated Provider
```

rather than reopening model execution semantics.

The exact roadmap remains a separate decision.

## Non-Goals

This change does not:

- add a new Runtime abstraction
- define Tensor Parallel
- define collectives
- define accelerator Provider contracts
- require high performance
- require production-scale Qwen
- require remote model acquisition
- define release v0.2
- define every implementation detail
- require exactly twenty PRs
- supersede existing security/release policy

## Impact

This change converts Magnetar from an architecture-design program into an
implementation program.

The project state becomes:

```text
DESIGN
  complete enough for first vertical slice

        |
        v

IMPLEMENTATION CUT
  this change

        |
        v

FIRST REAL MODEL
  Qwen WASM + Reference CPU

        |
        v

E2E CONFORMANCE

        |
        v

BASELINE STABILIZED
```