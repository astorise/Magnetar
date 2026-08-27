# operator Specification

## Purpose
This specification defines portable operator identity, attributes, validation, dtype/layout/shape constraints, and graph compatibility.
## Requirements
### Requirement: Operator

Magnetar SHALL define Operator as a portable semantic operation.

Operators SHALL describe what computation is performed, not how a Provider
implements it.

#### Scenario: Matmul operator

Given a graph contains a matmul operation

When Runtime validates it

Then it is represented as a portable Operator.

---

### Requirement: Operator Is Not Kernel

An Operator SHALL NOT be a Provider-specific kernel.

#### Scenario: CUDA matmul

Given a CUDA Provider has an optimized matmul kernel

When Runtime represents the graph

Then the graph still contains the matmul Operator, not a CUDA-specific operator.

---

### Requirement: Operator Identity

Operator identity SHALL be stable and versioned.

Identity SHOULD include namespace, name, semantic version, family, input/output
contract, attribute schema, shape rules, dtype rules, layout rules, memory
behavior, determinism metadata, and error behavior.

#### Scenario: Unsupported version

Given a graph references `magnetar:operator/attention@99`

When Runtime validates it

Then validation fails with operator-version-unsupported.

---

### Requirement: Operator Catalog

Magnetar SHALL define a versioned operator catalog.

Initial families SHOULD include tensor, linear-algebra, normalization,
position-encoding, attention, activation, quantization, layout,
sampling-support, and control.

#### Scenario: Unknown operator

Given a graph references an operator absent from the catalog

When Runtime validates it

Then validation fails with operator-not-found.

---

### Requirement: Operator Attributes

Operator attributes SHALL be validated.

Attributes SHALL NOT select Provider or Device directly.

#### Scenario: Provider attribute

Given an operator attribute attempts to force Provider `cuda`

When validation runs

Then Runtime rejects the attribute or treats it as non-authoritative invalid
metadata.

---

### Requirement: Shape Contract

Operators SHALL define shape requirements and shape inference behavior where
possible.

#### Scenario: Matmul shape mismatch

Given matmul inputs have incompatible dimensions

When Runtime validates the operator

Then validation fails with shape-mismatch.

---

### Requirement: DType Contract

Operators SHALL define supported input, output, storage, compute, and
accumulation dtype behavior where relevant.

#### Scenario: Unsupported dtype

Given RMSNorm receives unsupported dtype

When Runtime validates the operator

Then validation fails with dtype-unsupported.

---

### Requirement: Layout Contract

Operators SHALL define supported tensor layouts.

Layout conversion SHALL be explicit.

#### Scenario: Layout unsupported

Given attention requires paged layout

And input uses incompatible contiguous layout

When planning runs

Then Runtime inserts explicit conversion or fails.

---

### Requirement: Memory Behavior

Operators SHALL declare memory behavior including reads, writes, mutation,
aliasing, workspace, in-place support, residency requirements, and streaming
behavior where relevant.

#### Scenario: Workspace required

Given an operator requires temporary workspace

When graph is planned

Then Runtime requests Memory Manager admission.

---

### Requirement: Resource Affinity Preservation

Operators SHALL preserve Resource Affinity and SHALL NOT cause silent movement.

#### Scenario: Affinity conflict

Given an input tensor is bound to Provider A

When operator execution is planned on Provider B

Then planning fails or inserts explicit authorized movement.

---

### Requirement: Attention Operator

Magnetar SHALL represent attention as an operator family.

Attention metadata SHOULD include causal mode, mask kind, head counts, head
dimension, sequence length, context length, KV cache usage, paged cache support,
position encoding dependency, dtype, and layout requirements.

#### Scenario: Attention with KV cache

Given decode attention uses KV cache

When graph validation runs

Then attention metadata includes KV cache dependency.

---

### Requirement: Paged Attention Metadata

Paged attention metadata SHALL be representable as an attention variant or attribute.

This change SHALL not require implementation.

#### Scenario: Paged metadata

Given KV cache uses paged layout

When attention operator is represented

Then paged metadata can be expressed.

---

### Requirement: RoPE Operator

Rotary position embedding SHALL be represented as a position-encoding operator
or explicit attention attribute.

#### Scenario: RoPE metadata

Given model architecture uses RoPE

When graph is created

Then RoPE base, scale, dimension, and position mode are represented.

---

### Requirement: Normalization Operators

Magnetar SHALL define RMSNorm and LayerNorm operator families.

#### Scenario: RMSNorm

Given graph uses RMSNorm

When Runtime validates it

Then epsilon, normalized dimension, dtype, and accumulation behavior are
validated.

---

### Requirement: Quantization Operators

Quantization-related operators SHALL represent quantization behavior explicitly
where used.

#### Scenario: Dequantize before matmul

Given quantized weights require dequantization

When graph planning runs

Then dequantize behavior is explicit in graph or plan.

---

### Requirement: Adapter-Aware Operators

Adapter effects SHALL be represented explicitly through additional operators,
modified graph paths, or fused adapter metadata.

#### Scenario: LoRA overlay

Given LoRA adapter is active

When graph is created

Then adapter overlay is represented explicitly.

---

### Requirement: KV-Cache Operators

Operators that consume or produce KV cache SHALL declare that behavior
explicitly.

#### Scenario: KV append

Given decode step appends KV state

When graph is planned

Then KV cache append behavior is explicit.

---

### Requirement: Sampling Helper Operators

Sampling helper operators SHALL be representable for Provider-assisted or Device-resident
processing.

Sampling semantics remain owned by the Sampling Contract.

#### Scenario: Top-k helper

Given Provider-assisted top-k helper is represented in graph

When token selection occurs

Then Sampling Contract still owns final selection semantics.

---

### Requirement: Operator Determinism Metadata

Operators SHALL declare determinism metadata.

#### Scenario: Non-deterministic reduction

Given an operator uses a non-deterministic parallel reduction

When Generation requests deterministic mode

Then Runtime surfaces determinism limitation.

---

### Requirement: Operator Error Categories

Operator failures SHALL use structured error categories.

#### Scenario: Workspace unavailable

Given operator workspace cannot be allocated

When planning runs

Then Runtime returns workspace-unavailable or memory admission failure.

---

### Requirement: Browser-Compatible Operators

Operator Contract SHALL be platform-neutral and SHALL not require Wasmtime or
native Provider loading.

#### Scenario: Browser unsupported layout

Given browser target does not support a required layout

When graph validation runs

Then Runtime returns browser-feature-unsupported or layout-unsupported.

---

### Requirement: Operator Observability

Runtime SHALL define operator observations for planning, execution, completion,
failure, conversion insertion, workspace request, and Resource Affinity conflict.

Observability SHALL not expose raw tensor values by default.

#### Scenario: Operator failed

Given an operator execution fails

When observability records it

Then Runtime emits redacted operator error metadata.

### Requirement: Operators Are Implemented By Kernels

Operators SHALL be implementable by one or more Kernels.

#### Scenario: Multiple matmul kernels

Given CPU and CUDA Providers both advertise matmul Kernels

When Runtime plans matmul execution

Then each Kernel is considered an implementation of the matmul Operator.

---

### Requirement: Operator Semantics Constrain Kernels

A Kernel implementing an Operator SHALL preserve the Operator's declared
semantics.

#### Scenario: Approximate operator behavior

Given a Kernel changes observable Operator behavior beyond allowed tolerance

When conformance validates it

Then the Kernel fails conformance.

---

### Requirement: Operator Metadata Feeds Kernel Compatibility

Operator metadata SHALL be used to validate Kernel compatibility, including
attributes, shape rules, dtype rules, layout rules, memory behavior, and
determinism metadata.

#### Scenario: Attention attributes

Given attention Operator requires causal mode

When Runtime selects a Kernel

Then candidate Kernel must support that causal mode.

### Requirement: Operator Invocation Drives Kernel Selection

Operator invocation metadata SHALL drive Kernel Registry candidate lookup and
selection.

#### Scenario: Matmul invocation

Given graph planning produces a matmul Operator invocation

When Kernel Registry is queried

Then candidate Kernels implementing matmul are considered.

---

### Requirement: Operator Semantics Constrain Dispatch

Kernel Dispatch SHALL preserve Operator semantics.

#### Scenario: Fused candidate

Given a fused Kernel is selected

When dispatch is planned

Then Runtime validates that Operator semantics remain preserved.

### Requirement: Model Component Declares Operator Requirements

Model Components SHALL declare portable Operator requirements for architecture
execution.

#### Scenario: Qwen operators

Given Qwen Model Component supports decode

When Runtime inspects requirements

Then it sees portable Operator IDs such as attention, matmul, rope, rmsnorm, and
activation.

---

### Requirement: Operator Requirements Are Not Kernel Requirements

Model Component Operator requirements SHALL not be authoritative Provider Kernel
selection.

#### Scenario: Kernel name declared

Given a Model Component declares a Provider-specific Kernel name as required

When Runtime validates it

Then Runtime rejects it or treats it as non-authoritative invalid metadata.

---

### Requirement: Reference CPU Validates Operator Semantics

Reference CPU Provider SHALL provide baseline behavior for supported Operators.

#### Scenario: Operator conformance

Given an Operator has Reference CPU implementation

When conformance runs

Then outputs are compared against expected Operator semantics.

---

### Requirement: Unsupported Operators Are Not Assumed

Runtime SHALL not assume Reference CPU supports an Operator unless a Kernel is
advertised.

#### Scenario: GELU unsupported

Given no Reference CPU GELU Kernel is advertised

When graph requires GELU

Then Runtime reports missing Kernel or uses explicit fallback according to
policy.

### Requirement: Operators Have Implementation Scope Classification

Operators SHALL be classifiable by first implementation scope.

#### Scenario: Classify attention

Given attention Operator metadata is inspected

When first scope is active

Then attention is classified as required-now or as the specific supported
attention baseline.

---

### Requirement: Required Operators Have Conformance Fixtures

Operators in required-now scope SHALL have conformance fixtures.

#### Scenario: Matmul fixture

Given matmul is required-now

When conformance runs

Then matmul fixtures are available.

---

### Requirement: Placeholder Operators Do Not Imply Kernel Availability

A placeholder Operator SHALL not imply that a Kernel exists.

#### Scenario: Paged attention placeholder

Given paged-attention Operator identity exists

When Kernel Registry is queried

Then no Kernel is assumed unless advertised.

### Requirement: Operators Use Tensor Contracts

Operators SHALL consume and produce Tensor Descriptors or Tensor Resources using the Tensor Resource and Layout Contract.

#### Scenario: Operator input validation

Given Operator receives Tensor Resource input

When validation runs

Then shape, dtype, layout, memory behavior, aliasing, and Resource Affinity are validated.

---

### Requirement: Operators Do Not Require Raw Tensor Pointers

Operator semantics SHALL be expressed without raw memory pointers.

#### Scenario: Matmul semantics

Given matmul Operator is defined

When Component inspects it

Then it sees Tensor metadata requirements, not raw pointer requirements.

---

### Requirement: Operator Layout Requirements Are Explicit

Operator layout requirements SHALL be explicit and validated.

#### Scenario: Layout mismatch

Given Operator requires contiguous input

When input is strided

Then Runtime rejects or plans explicit layout conversion.

---

### Requirement: Qwen Baseline Operators Are First-Scope Operators

Qwen baseline SHALL use first-scope required Operators unless explicit policy
allows additional implemented Operators.

#### Scenario: Required operator set

Given Qwen baseline graph is inspected

When operator scope validation runs

Then all Operators are required-now or explicitly supported.

---

### Requirement: Qwen Operator Requirements Are Portable

Qwen operator requirements SHALL reference portable Operator IDs, not
Provider-specific Kernel names.

#### Scenario: Kernel-specific requirement

Given Qwen Component declares `cuda.flash_attention_v2`

When Runtime validates requirements

Then validation fails or marks it non-authoritative invalid metadata.

---

### Requirement: E2E Exercises Required Operators

E2E conformance SHALL exercise required-now Operators for the first decoder
baseline.

#### Scenario: Operator coverage

Given E2E success path completes

When report is generated

Then required operator coverage is recorded.

---

### Requirement: E2E Fails Missing Operator Coverage

E2E conformance SHALL report missing required operator coverage.

#### Scenario: Missing RoPE coverage

Given fixture path does not exercise RoPE

When operator coverage is required

Then E2E reports missing coverage.

---

### Requirement: First Operator Scope Implemented Before Qwen Baseline

The first operator implementation scope SHALL be implemented before Qwen
baseline graph execution.

#### Scenario: Qwen graph validates

Given Qwen graph uses attention and RMSNorm

When graph validation runs

Then required-now operator metadata exists.

---

### Requirement: Operator Fixtures Support CPU Baseline

Required-now Operators SHALL have fixtures usable by Reference CPU conformance.

#### Scenario: Softmax fixture

Given softmax is required-now

When CPU conformance runs

Then softmax fixture exists.

---

### Requirement: Optimized Providers Preserve Operator Semantics

Optimized Providers SHALL preserve portable Operator semantics.

#### Scenario: Optimized softmax

Given optimized softmax Kernel is selected

When output is compared to reference fixture

Then output matches within declared tolerance.

---

### Requirement: Advanced Operator Variants Are Explicit

Advanced Operator variants such as flash attention or paged attention SHALL be
explicit variants or graph fragments, not hidden substitutions.

#### Scenario: Hidden flash attention

Given Provider silently replaces attention with unsupported flash semantics

When conformance validates execution

Then conformance fails.

### Requirement: First Operator Scope Release Gate

First operator scope conformance SHALL be required for `v0.1`.

#### Scenario: Required operator missing

Given required operator coverage is missing

When release validation runs

Then stable release is blocked.

---

### Requirement: Operator Compatibility Status

Release matrix SHALL document Operator catalog compatibility status.

#### Scenario: Operator status

Given release notes are generated

When Operator catalog is listed

Then its status is declared.

