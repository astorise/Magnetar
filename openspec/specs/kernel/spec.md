# kernel Specification

## Purpose
This specification defines kernel metadata, validation, precision/determinism policy, memory workspace requirements, and Provider execution contracts.
## Requirements
### Requirement: Kernel

Magnetar SHALL define Kernel as a concrete implementation of an Operator for a
Provider and compatible Device context.

#### Scenario: Matmul kernel

Given a Provider implements matmul for FP16 on CUDA

When Runtime reads Provider metadata

Then the implementation is represented as a Kernel implementing the matmul
Operator.

---

### Requirement: Kernel Is Not Operator

A Kernel SHALL not define portable semantics.

Portable semantics SHALL remain owned by the Operator Contract.

#### Scenario: Kernel implements attention

Given a Provider advertises a flash attention Kernel

When Runtime validates it

Then the Kernel declares compatibility with the portable attention Operator.

---

### Requirement: Kernel Is Not Provider

A Kernel SHALL belong to or be exposed by a Provider but SHALL not itself be a
Provider.

#### Scenario: Provider advertises kernels

Given a CUDA Provider advertises matmul and attention Kernels

When Runtime registers them

Then the Provider remains the execution extension and Kernels remain
implementation metadata.

---

### Requirement: Kernel Is Not Model Architecture

Kernels SHALL not be selected primarily by model family.

#### Scenario: Qwen attention

Given Qwen model uses attention attributes

When Runtime plans execution

Then Runtime selects an attention Kernel compatible with those attributes, not a
Qwen-specific Provider.

---

### Requirement: Kernel Identity

Kernel identity SHALL be stable within its Provider and SHALL not expose raw
function pointers.

#### Scenario: Kernel status

Given Runtime reports Kernel metadata

When a caller inspects it

Then the report includes stable identity and not native function addresses.

---

### Requirement: Kernel Advertisement

Providers SHALL advertise Kernel metadata through Runtime-readable
advertisements.

Runtime SHALL not rely on undocumented Kernel behavior.

#### Scenario: Missing dtype advertisement

Given a Kernel does not advertise BF16 support

When BF16 execution is planned

Then Runtime does not assume BF16 support.

---

### Requirement: Kernel Operator Compatibility

Runtime SHALL validate Kernel compatibility with Operator invocation before
dispatch.

#### Scenario: Operator mismatch

Given a Kernel implements matmul

When Runtime attempts to dispatch attention to it

Then dispatch is rejected with kernel-Operator-mismatch.

---

### Requirement: Kernel Shape Constraints

A Kernel SHALL declare shape constraints.

#### Scenario: Sequence too long

Given a Kernel supports max sequence length 4096

When invocation requires 8192

Then Runtime rejects the Kernel for that invocation.

---

### Requirement: Kernel DType Constraints

A Kernel SHALL declare supported dtype combinations.

#### Scenario: Unsupported dtype

Given a Kernel supports FP16 input only

When invocation uses INT8 input

Then Runtime rejects dispatch or plans explicit conversion according to policy.

---

### Requirement: Kernel Layout Constraints

A Kernel SHALL declare supported layouts.

Provider-owned opaque layouts SHALL not leak into portable Component APIs.

#### Scenario: Layout mismatch

Given a Kernel requires blocked layout

And input is contiguous

When Runtime plans execution

Then Runtime inserts explicit layout conversion or rejects execution.

---

### Requirement: Kernel Memory Class Constraints

A Kernel SHALL declare supported memory classes.

#### Scenario: Device memory required

Given a Kernel requires Device memory

When input is host-only

Then Runtime performs explicit movement or rejects dispatch according to policy.

---

### Requirement: Kernel Workspace Requirements

A Kernel SHALL declare workspace requirements.

Workspace allocation SHALL be performed through Memory Manager.

#### Scenario: Workspace unavailable

Given Kernel workspace cannot be allocated

When Runtime plans dispatch

Then dispatch fails or falls back according to policy.

---

### Requirement: Kernel Aliasing Behavior

A Kernel SHALL declare input/output aliasing and mutation behavior.

#### Scenario: In-place unsupported

Given an invocation aliases input and output

And Kernel does not support in-place execution

When Runtime validates dispatch

Then dispatch is rejected.

---

### Requirement: Kernel Preserves Resource Affinity

Kernel execution SHALL preserve Resource Affinity.

Runtime SHALL not silently move data to satisfy Kernel requirements.

#### Scenario: Affinity conflict

Given input tensor is bound to Device A

And Kernel can only run on Device B

When dispatch is planned

Then Runtime inserts explicit authorized movement or rejects dispatch.

---

### Requirement: Kernel Execution Mode

A Kernel SHALL declare execution mode.

#### Scenario: Asynchronous kernel

Given a Kernel is asynchronous

When Runtime dispatches it

Then Runtime tracks completion, cancellation support, and memory lifetime
accordingly.

---

### Requirement: Kernel Cancellation Metadata

A Kernel SHALL declare cancellation support.

Runtime SHALL not assume cancellation during execution.

#### Scenario: Cancellation unsupported

Given Kernel supports only before-dispatch cancellation

When cancellation is requested during execution

Then Runtime reports cancellation limitation according to policy.

---

### Requirement: Kernel Determinism Metadata

A Kernel SHALL declare determinism metadata.

#### Scenario: Deterministic requested

Given deterministic generation is requested

When candidate Kernel is nondeterministic

Then Runtime rejects it or reports determinism unsupported according to policy.

---

### Requirement: Kernel Precision Metadata

A Kernel SHALL declare precision and tolerance metadata.

#### Scenario: Approximate math

Given a Kernel uses approximate math

When precision policy requires exact behavior

Then Runtime rejects it or selects another Kernel.

---

### Requirement: Fused Kernel Semantics

Fused Kernels SHALL declare the Operator or Operator group they implement and
SHALL preserve graph semantics.

#### Scenario: Fused RMSNorm Matmul

Given a Kernel fuses RMSNorm and Matmul

When Runtime considers it

Then Runtime validates that fusion preserves graph semantics.

---

### Requirement: Adapter-Aware Kernel Metadata

Adapter-aware Kernels SHALL declare adapter method, rank, dtype, target module,
and execution strategy compatibility.

#### Scenario: LoRA rank unsupported

Given active LoRA rank is 64

And Kernel supports rank up to 32

When Runtime validates Kernel compatibility

Then Kernel is rejected.

---

### Requirement: KV-Cache-Aware Kernel Metadata

KV-cache-aware Kernels SHALL declare cache layout, paged cache, append/read
behavior, cache dtype, memory class, and Resource Affinity constraints.

#### Scenario: Paged cache unsupported

Given KV cache uses paged layout

And Kernel does not support paged cache

When Runtime validates dispatch

Then Kernel is rejected.

---

### Requirement: Prefix Cache Boundary Support

Kernels SHALL support adjusted sequence/context metadata when Prefix Cache reuse
changes prefill boundaries where relevant.

#### Scenario: Prefix hit shortens prefill

Given Prefix Cache reuses 1000 tokens

When Kernel is dispatched for remaining prefill

Then invocation shape metadata reflects the adjusted boundary.

---

### Requirement: Batched Kernel Metadata

Batched Kernels SHALL declare batch size, active sequence, total token, ragged
batch, paged KV cache, output mapping, and slot compatibility metadata where
supported.

#### Scenario: Batch too large

Given Kernel supports batch size 8

When Scheduler proposes batch size 16

Then Runtime rejects that Kernel for the batch.

---

### Requirement: Browser-Compatible Kernel Contract

Kernel Contract SHALL be platform-neutral and SHALL not require Wasmtime or
native Provider loading.

#### Scenario: Browser target

Given browser target lacks native Kernel path

When Runtime plans Kernel dispatch

Then Runtime uses browser-compatible Kernel metadata or returns
kernel-browser-feature-unsupported.

---

### Requirement: Runtime-Created Kernel Invocation

Kernel Invocation SHALL be created by Runtime.

Components SHALL NOT create raw Provider Kernel invocations directly.

#### Scenario: Component attempts kernel call

Given a Component attempts to call a Provider Kernel directly

When Runtime validates authority

Then Runtime denies the direct call.

---

### Requirement: Kernel Result

Kernel execution SHALL return structured results and SHALL not expose raw
Provider handles or memory pointers.

#### Scenario: Kernel succeeds

Given Kernel execution completes

When Runtime receives the result

Then it updates resource metadata and returns stable output readiness.

---

### Requirement: Kernel Error Categories

Kernel failures SHALL use structured error categories.

#### Scenario: Provider saturated

Given Kernel dispatch fails because Provider is saturated

When Runtime reports the error

Then it returns kernel-Provider-saturated or equivalent mapped error.

---

### Requirement: Kernel Conformance

Kernels SHALL be subject to conformance testing tied to Operator semantics and
Kernel metadata.

#### Scenario: Conformance failure

Given a Kernel produces incorrect matmul output beyond tolerance

When conformance runs

Then Kernel fails the relevant conformance profile.

---

### Requirement: Kernel Fallback Is Explicit

Kernel fallback SHALL be explicit and SHALL not silently violate policy.

#### Scenario: Host fallback

Given Device Kernel is unavailable

When host fallback is considered

Then Runtime validates Resource Affinity, memory, dtype, layout, and policy
before using it.

---

### Requirement: Kernel Security Boundary

Kernel execution SHALL not expose raw memory, model weights, prompts, KV cache,
Provider handles, or Device handles to Components or clients.

#### Scenario: Kernel diagnostics

Given Kernel diagnostics are requested

When Runtime returns them

Then diagnostics are redacted and stable.

---

### Requirement: Kernel Observability

Runtime SHALL emit Kernel observations for advertisement, invocation, dispatch,
completion, failure, workspace, cancellation, fallback, conformance, Resource
Affinity conflicts, determinism, and precision diagnostics.

Observability SHALL not expose raw tensor values, prompts, weights, KV cache
contents, handles, or memory pointers by default.

#### Scenario: Kernel dispatch failed

Given Kernel dispatch fails

When observability records it

Then Runtime emits stable redacted Kernel error metadata.

### Requirement: Kernels Are Registered Through Kernel Registry

Kernel advertisements SHALL enter Runtime use through Kernel Registry
validation.

#### Scenario: Provider advertises Kernel

Given Provider advertises a Kernel

When Runtime accepts the advertisement

Then the Kernel becomes available as a registry candidate.

---

### Requirement: Kernels Are Dispatched Through Runtime

Kernels SHALL be dispatched through Runtime-created Kernel Dispatch Plans and
Kernel Invocations.

#### Scenario: Dispatch Kernel

Given a Kernel is selected

When execution begins

Then Runtime dispatches it through the owning Provider.

---

### Requirement: Kernel Metadata Supports Registry Selection

Kernel metadata SHALL be sufficient for Registry filtering, ranking, fallback,
dispatch, and conformance gating.

#### Scenario: Missing shape metadata

Given a Kernel advertisement lacks required shape constraints

When Runtime validates it

Then the advertisement is rejected or marked unusable.

---

### Requirement: Kernel Dispatch Is Revalidated

A selected Kernel SHALL be revalidated before dispatch.

#### Scenario: Provider saturated

Given Kernel was selected while Provider pressure was low

But Provider becomes saturated before dispatch

When revalidation runs

Then Runtime delays, falls back, or rejects according to policy.

---

### Requirement: Reference CPU Kernels Implement Operators

Reference CPU Kernels SHALL implement portable Operators according to the Kernel
Contract.

#### Scenario: CPU RMSNorm Kernel

Given Reference CPU Provider advertises RMSNorm

When Runtime validates the Kernel

Then it is tied to the portable RMSNorm Operator.

---

### Requirement: Reference CPU Kernels Prioritize Correctness

Reference CPU Kernels SHALL prioritize correctness and conformance over
performance.

#### Scenario: Slow attention

Given CPU attention is quadratic and slow

When it matches Operator semantics

Then it is acceptable for reference execution.

---

### Requirement: Reference CPU Kernels Declare Limitations

Reference CPU Kernels SHALL declare unsupported dtype, layout, shape, memory
class, batching, cancellation, and precision limitations.

#### Scenario: Unsupported paged cache

Given CPU attention does not support paged KV cache

When Kernel metadata is advertised

Then paged cache support is absent or explicitly unsupported.

### Requirement: Kernels Declare First Scope Coverage

Kernel metadata SHALL indicate whether the Kernel participates in first
operator implementation scope.

#### Scenario: CPU kernel in scope

Given Reference CPU matmul kernel is advertised

When Registry records it

Then metadata may mark it as first-scope capable.

---

### Requirement: Placeholder Kernels Require Explicit Status

If a Kernel advertisement corresponds to a placeholder Operator, Runtime SHALL
require explicit implemented status and conformance before use.

#### Scenario: Placeholder kernel

Given Provider advertises paged-attention

When first scope validates it

Then Runtime requires concrete support metadata and conformance status.

### Requirement: Kernels Receive Runtime Tensor Resource References

Kernels SHALL receive Runtime-created resource references rather than public raw pointers.

#### Scenario: Dispatch kernel

Given Kernel Invocation is created

When Provider receives it

Then it receives validated resource references and metadata.

---

### Requirement: Kernel Dispatch Validates Tensor Metadata

Kernel Dispatch SHALL validate Tensor Resource shape, dtype, layout, memory class, readiness, aliasing, mutability, and Resource Affinity before execution.

#### Scenario: Tensor not ready

Given input Tensor Resource is pending transfer

When Kernel dispatch validates inputs

Then dispatch is delayed, rejected, or replanned according to policy.

---

### Requirement: Kernel Results Update Tensor Metadata

Kernel Results SHALL update Tensor Resource readiness, residency, Resource Affinity, aliasing, and lifecycle metadata where relevant.

#### Scenario: Output tensor produced

Given Kernel writes output

When dispatch completes

Then Runtime marks output Tensor Resource ready.

---

### Requirement: Fused Kernels Declare Semantic Equivalence

Post-baseline fused Kernels SHALL declare semantic equivalence to portable
Operator sequences or graph fragments.

#### Scenario: Fused MLP

Given Provider advertises fused MLP Kernel

When Runtime validates it

Then metadata identifies the equivalent portable Operator sequence.

---

### Requirement: Advanced Kernels Declare Specialized Requirements

Advanced Kernels SHALL declare dtype, layout, memory class, precision,
determinism, and Resource Affinity requirements.

#### Scenario: Flash attention kernel

Given Provider advertises flash attention

When Kernel metadata is inspected

Then required layout, dtype, memory class, and precision tolerance are explicit.

---

### Requirement: Kernel May Be Artifact-Backed

An artifact-backed Kernel SHALL implement the same portable Operator
semantics as a statically defined Kernel. A Kernel implementation MAY be
backed by a Kernel Artifact lifecycle.

#### Scenario: Generated MatMul

Given generated MatMul artifact is prepared

When Kernel Registry advertises it

Then the prepared implementation remains a Kernel implementing portable
MatMul semantics.

---

### Requirement: Kernel Identity Is Separate From Prepared State

KernelId SHALL remain logical implementation identity and SHALL NOT be the
native prepared handle.

#### Scenario: Same Kernel prepared twice

Given same KernelId is prepared for two Devices

When Registry tracks them

Then each PreparedKernelId is distinct while KernelId semantics remain the
same.

---

### Requirement: Kernel Advertisement May Reference Artifact Metadata

Artifact metadata referenced by KernelAdvertisement SHALL NOT replace
KernelId as the authoritative logical identity. KernelAdvertisement MAY
reference artifact identity and preparation metadata.

#### Scenario: Generated kernel advertisement

Given generated kernel is advertised

When Registry evaluates it

Then artifact identity and build fingerprint may participate in selection.

---

### Requirement: Kernel Native State Remains Provider Private

Kernel contracts SHALL not expose Provider-native executable pointers.

#### Scenario: CUDA Kernel

Given CUDA Provider owns CUfunction

When Kernel metadata is returned

Then CUfunction address is absent.

---

### Requirement: Executable Kernel Uses Prepared State

Artifact-backed Kernel SHALL execute only through previously prepared Provider
state.

#### Scenario: Kernel dispatch

Given compatible Kernel has no PreparedKernelId

When dispatch runs

Then Runtime does not invoke source compilation through execution path.

---

### Requirement: Compilation Capability Is Not Kernel Semantics

Kernel compilation mechanism SHALL NOT change portable Operator semantics.

#### Scenario: Triton MatMul

Given MatMul is implemented through generated Triton

When Kernel is registered

Then it still implements the existing portable MatMul Operator contract.

---

### Requirement: Kernel Exposes Selection Metadata

Any KernelAdvertisement metadata SHALL accurately describe the Kernel's actual behavior, and MAY expose policy-relevant metadata such as performance, workspace, determinism and specialization.

#### Scenario: High-workspace Kernel

Given Kernel needs 128 MiB workspace

When selection evaluates memory profile

Then workspace requirement participates in decision.

---

### Requirement: Performance Metadata Does Not Define Semantics

Kernel performance metadata SHALL NOT change Operator semantics.

#### Scenario: Faster approximate kernel

Given approximation changes numerical contract

When semantics do not allow approximation

Then Kernel is incompatible regardless of benchmark.

---

### Requirement: Runtime-Relevant Variant Differences Require Distinct Candidate

Provider variants differing in Runtime-relevant semantics or constraints SHALL
be separately represented.

#### Scenario: Deterministic and nondeterministic implementations

Given Provider has both variants

When determinism differs

Then they SHALL be distinguishable candidates rather than invisible private
switch.

### Requirement: Required Operators Have Reference CPU Kernels

Every mandatory first-profile Operator SHALL have a Reference CPU Kernel path.

#### Scenario: RMSNorm dispatch

Given Qwen graph contains RMSNorm

When Registry resolves it

Then eligible Reference CPU Kernel exists.

### Requirement: Reference Kernels Are Correctness Baseline

First Reference CPU Kernels SHALL prioritize deterministic understandable
correctness over optimization.

#### Scenario: Scalar MatMul

Given unoptimized implementation is mathematically correct

When first-profile tests run

Then lack of SIMD does not fail conformance.

### Requirement: Kernels Are Selected Through Registry

Reference CPU Kernel SHALL not become an architecture-specific direct call from
Qwen execution.

#### Scenario: Attention executes

Given graph node is Attention

When Runtime executes

Then Kernel selection passes through Kernel Registry/Dispatch.

### Requirement: Prepared Kernel Lifecycle Is Used

Reference CPU Kernel execution SHALL participate in PreparedKernel contract.

#### Scenario: MatMul Plan binding

Given Kernel is selected

When Plan becomes ready

Then binding references opaque PreparedKernelId or equivalent prepared state.

