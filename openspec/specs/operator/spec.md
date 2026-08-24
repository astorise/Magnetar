# operator Specification

## Purpose
TBD - created by archiving change define-execution-graph-and-operator-contract. Update Purpose after archive.
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

