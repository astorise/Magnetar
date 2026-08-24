## ADDED Requirements
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
