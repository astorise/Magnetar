# operator-scope Specification

## Purpose
TBD - created by archiving change define-first-operator-implementation-scope. Update Purpose after archive.
## Requirements
### Requirement: First Operator Implementation Scope

Magnetar SHALL define a first operator implementation scope for the initial
executable inference baseline.

#### Scenario: Scope queried

Given Runtime needs first implementation metadata

When Operator Scope is queried

Then required-now, placeholder, unsupported, and future-optimized tiers are
available.

---

### Requirement: Required-Now Tier

Operators in the required-now tier SHALL have explicit semantics, validation,
Reference CPU coverage, and conformance fixtures.

#### Scenario: Required operator

Given `matmul` is required-now

When Reference CPU Provider initializes

Then a compatible matmul kernel is advertised or validation fails.

---

### Requirement: Placeholder Tier

Placeholder operators SHALL reserve identity without implying implementation.

#### Scenario: Placeholder operator

Given `paged-attention` is placeholder

When graph planning requires it

Then Runtime reports placeholder-only or unsupported unless policy allows a
non-executing path.

---

### Requirement: Explicitly Unsupported Tier

Explicitly unsupported operators SHALL fail with structured errors.

#### Scenario: Training operator

Given a graph contains a gradient operator

When Runtime validates first scope

Then Runtime rejects it as explicitly unsupported.

---

### Requirement: Future Optimized Tier

Future optimized operators SHALL not change portable semantics when introduced.

#### Scenario: Future flash attention

Given flash attention is later introduced

When it replaces attention path

Then Runtime validates semantic equivalence or explicit variant compatibility.

---

### Requirement: First Decoder Model Scope

The first operator scope SHALL support a minimal decoder-only transformer path.

#### Scenario: Decoder graph

Given a first decoder graph uses embedding, RMSNorm, matmul, RoPE, attention,
softmax, SiLU, add, mul, residual-add, and logits matmul

When graph planning runs

Then all operators are within first implementation scope.

---

### Requirement: Required CPU Coverage

Required-now operators SHALL have Reference CPU kernel coverage or fail
validation.

#### Scenario: Missing RMSNorm CPU kernel

Given RMSNorm is required-now

And Reference CPU Provider does not advertise RMSNorm

When first scope validation runs

Then validation fails with first-scope-kernel-missing.

---

### Requirement: No Silent Quantization

Quantized operators SHALL not be silently emulated.

#### Scenario: Quantized matmul

Given graph requires quantized-matmul

When first scope validates it

Then Runtime reports placeholder-only, unsupported, or missing kernel.

---

### Requirement: No Silent Fusion

The first implementation scope SHALL not require fused kernels.

#### Scenario: MLP graph

Given MLP uses matmul, SiLU, mul, and matmul

When graph planning runs

Then unfused execution is valid.

---

### Requirement: Initial DType Scope

Initial compute dtype SHALL prioritize f32.

Unsupported dtype compute SHALL fail or require explicit conversion.

#### Scenario: BF16 compute

Given BF16 compute is requested

And BF16 is placeholder

When planning runs

Then Runtime rejects or inserts explicit conversion if policy allows.

---

### Requirement: Initial Layout Scope

Initial layout support SHALL prioritize contiguous layout.

Unsupported layouts SHALL fail or require explicit conversion.

#### Scenario: Blocked layout

Given blocked layout is required

When first scope validates layout

Then Runtime rejects it unless explicit conversion exists.

---

### Requirement: Shape Scope

The first scope SHALL validate ranks, batch, sequence, hidden size, head count,
KV head count, head dimension, intermediate size, vocabulary size, broadcasting,
and matmul compatibility where relevant.

#### Scenario: Matmul mismatch

Given incompatible matmul dimensions

When first scope validation runs

Then Runtime rejects with shape error.

---

### Requirement: Attention Scope

Initial attention SHALL support only explicitly implemented variants.

Unsupported variants SHALL fail.

#### Scenario: Flash attention requested

Given graph requires flash attention

When first scope validates it

Then Runtime rejects it as unsupported or future optimized.

---

### Requirement: RoPE Scope

Initial RoPE SHALL support one explicit baseline mode and reject unsupported
variants.

#### Scenario: Dynamic RoPE unsupported

Given model requires unsupported dynamic RoPE scaling

When first scope validates it

Then Runtime rejects with attribute unsupported.

---

### Requirement: MLP Scope

Initial MLP SHALL be expressible with unfused required-now operators.

#### Scenario: Gated MLP

Given graph uses matmul, SiLU, mul, and matmul

When validation runs

Then MLP path is within scope.

---

### Requirement: Logits Projection Scope

Initial logits projection MAY use matmul and SHALL not require
Provider-assisted sampling.

#### Scenario: Logits projection

Given decoder output is projected to vocabulary logits

When graph planning runs

Then matmul may implement logits projection.

---

### Requirement: Model Component Scope Compatibility

Model Components used in the first baseline SHALL require only operators within
the first scope.

#### Scenario: Unsupported operator required

Given a Model Component requires MoE dispatch

When first baseline validates it

Then Runtime rejects the Component path.

---

### Requirement: First Scope Conformance

Each required-now operator SHALL have conformance fixtures.

#### Scenario: Conformance missing

Given `rope` is required-now

And no conformance fixture exists

When first scope conformance runs

Then conformance reports missing fixture.

---

### Requirement: First Scope Error Categories

First scope failures SHALL use structured error categories.

#### Scenario: Placeholder required at runtime

Given graph requires placeholder-only operator

When execution is requested

Then Runtime returns operator-placeholder-only or first-scope-kernel-missing.

---

### Requirement: First Scope Observability

Runtime SHALL emit observations for accepted, rejected, placeholder,
unsupported, missing kernel, unsupported dtype/layout/shape, and conformance
status.

Observability SHALL not expose raw tensor values, prompts, weights, cache
contents, handles, or memory pointers by default.

#### Scenario: Operator rejected

Given graph contains unsupported operator

When Runtime rejects it

Then observability records redacted operator-scope rejection metadata.

