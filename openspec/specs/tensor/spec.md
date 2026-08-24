# tensor Specification

## Purpose
TBD - created by archiving change define-tensor-resource-and-layout-contract. Update Purpose after archive.
## Requirements
### Requirement: Tensor Descriptor

Magnetar SHALL define Tensor Descriptor as portable tensor metadata that does not imply allocation.

#### Scenario: Descriptor only

Given an Operator declares output metadata

When Runtime creates a Tensor Descriptor

Then no storage allocation is implied by the descriptor alone.

---

### Requirement: Tensor Resource

Magnetar SHALL define Tensor Resource as Runtime-managed tensor storage or Provider-owned opaque tensor storage with Runtime-visible metadata.

#### Scenario: Tensor allocated

Given Memory Manager allocates output tensor storage

When allocation succeeds

Then Runtime creates or updates a Tensor Resource.

---

### Requirement: Tensor Resource Identity

Tensor Resource IDs SHALL be Runtime-issued and opaque.

They SHALL NOT encode raw pointers, Provider handles, Device handles, allocation addresses, file paths, secrets, or prompt data.

#### Scenario: Inspect tensor ID

Given a Tensor Resource ID is logged

When inspected

Then it does not reveal memory address or Provider handle.

---

### Requirement: Tensor Lifecycle

Tensor Resources SHALL have lifecycle state.

States SHOULD include declared, planned, allocating, ready, in-use, view, mutating, released, evicted, invalid, and failed.

#### Scenario: Output ready

Given Kernel execution writes output

When Runtime updates resource state

Then Tensor Resource lifecycle may become ready.

---

### Requirement: Tensor Readiness

Tensor readiness SHALL be distinct from lifecycle.

#### Scenario: Pending transfer

Given Tensor Resource exists

But host transfer is pending

When Kernel requires host-ready input

Then Runtime rejects or waits according to policy.

---

### Requirement: Tensor Shape

Tensor Shape SHALL be explicit and validated before dispatch where possible.

#### Scenario: Rank mismatch

Given Operator requires rank 2

And Tensor Resource has rank 3

When validation runs

Then Runtime returns tensor-rank-unsupported or tensor-shape-mismatch.

---

### Requirement: Tensor DType

Tensor DType metadata SHALL distinguish storage, compute, accumulation, output, index, and mask dtype where relevant.

#### Scenario: Silent conversion forbidden

Given input storage dtype is f16

And Kernel requires f32

When planning runs

Then Runtime inserts explicit dtype conversion or rejects execution.

---

### Requirement: Tensor Layout

Tensor Layout SHALL be explicit.

Unsupported layouts SHALL fail with structured errors or explicit conversion plans.

#### Scenario: Unsupported layout

Given Tensor Resource has blocked layout

And selected Kernel supports only contiguous layout

When validation runs

Then Runtime rejects or plans explicit layout conversion.

---

### Requirement: Contiguous Layout

Contiguous layout SHALL describe dense storage with explicit dimension order where needed.

#### Scenario: CPU contiguous tensor

Given Reference CPU Kernel requires contiguous input

When Tensor Resource is contiguous host memory

Then validation may succeed.

---

### Requirement: Strided Layout

Strided layout SHALL describe explicit strides and offset.

#### Scenario: Unsupported strided tensor

Given first scope does not support strided layout

When graph planning receives strided input

Then Runtime rejects or plans explicit materialization.

---

### Requirement: Paged Layout

Paged layout SHALL represent page/block-based storage without exposing raw page pointers.

#### Scenario: Paged KV tensor

Given KV cache uses paged layout metadata

When Runtime reports tensor metadata

Then page pointers are not exposed.

---

### Requirement: Packed Quantized Layout

Packed quantized layout SHALL represent quantized packed storage with explicit metadata.

#### Scenario: Quantized layout unsupported

Given packed quantized layout is encountered in first scope

When planning runs

Then Runtime rejects or requires explicit dequantization path.

---

### Requirement: Provider-Owned Opaque Layout

Provider-owned opaque layout SHALL not expose native internals to Components or clients.

#### Scenario: Opaque tensor

Given Provider owns opaque tensor storage

When Runtime reports tensor metadata

Then it reports stable metadata and not native handles.

---

### Requirement: Tensor View

Tensor Views SHALL be Runtime-authorized views over Tensor Resources and SHALL not outlive their base resource.

#### Scenario: Base released

Given a Tensor View references base tensor T

When T is released

Then the view becomes invalid or unavailable.

---

### Requirement: Aliasing

Tensor aliasing SHALL be explicit and validated before Kernel dispatch.

#### Scenario: In-place mutation denied

Given input and output alias

And Kernel does not support in-place mutation

When dispatch is validated

Then Runtime rejects it with tensor-aliasing-violation.

---

### Requirement: Mutability

Tensor mutability SHALL be explicit.

A Kernel SHALL not mutate input unless mutation is declared and allowed.

#### Scenario: Immutable input

Given Tensor Resource is immutable

When Kernel requests mutation

Then Runtime rejects dispatch.

---

### Requirement: Tensor Memory Class

Tensor Resources SHALL declare memory class.

#### Scenario: Host-only Kernel

Given Kernel supports host memory only

When Tensor Resource is device-only

Then Runtime rejects or plans explicit movement.

---

### Requirement: Tensor Residency

Tensor Resource residency SHALL be tracked by Memory Manager.

#### Scenario: Tensor on Device

Given Tensor Resource resides on Device A

When Runtime reports metadata

Then residency and Resource Affinity reflect Device A without exposing handle.

---

### Requirement: Resource Affinity Is Runtime-Derived

Tensor Resource Affinity SHALL be Runtime-derived and authoritative.

A caller SHALL NOT forge Provider or Device affinity.

#### Scenario: Forged affinity

Given caller claims Tensor Resource is on Device A

When Runtime validates resource metadata

Then caller metadata is ignored or rejected.

---

### Requirement: Tensor Size Accounting

Runtime SHALL compute or conservatively estimate tensor size from shape, dtype, layout, and packing metadata.

#### Scenario: Unknown size

Given packed layout lacks enough size metadata

When Memory Manager evaluates it

Then admission is conservative or rejected.

---

### Requirement: Tensor Conversion Is Explicit

DType, layout, memory movement, device transfer, host staging, opaque materialization, quantization, and dequantization SHALL be explicit operations or plans.

#### Scenario: Host staging forbidden

Given CPU fallback would require host staging

And policy forbids host staging

When planning runs

Then Runtime rejects fallback.

---

### Requirement: Tensor Materialization Is Runtime-Controlled

Tensor materialization SHALL be tracked by Runtime and Memory Manager.

#### Scenario: Materialize weights

Given model weights are loaded from Model Artifact

When tensors are materialized

Then Runtime creates Tensor Resources tracked by Memory Manager.

---

### Requirement: Component Tensor Access Boundary

Components SHALL not access raw tensor storage by default.

#### Scenario: Component requests pointer

Given a Component requests a raw Tensor pointer

When Runtime authorizes access

Then access is denied.

---

### Requirement: Runtime Tensor APIs Are Metadata-Safe

Runtime Tensor APIs SHALL expose stable metadata and controlled resource references only.

#### Scenario: Tensor status

Given caller requests Tensor Resource status

When Runtime responds

Then no raw pointer, handle, prompt, model weight, or KV cache content is included.

---

### Requirement: Tensor Error Categories

Tensor failures SHALL use structured error categories.

#### Scenario: Released tensor

Given Tensor Resource was released

When Kernel tries to use it

Then Runtime returns tensor-resource-released.

---

### Requirement: Tensor Observability

Runtime SHOULD emit Tensor observations for descriptor creation, planning, allocation, readiness, views, usage, mutation, conversion, transfer, release, eviction, invalidation, aliasing violation, and Resource Affinity conflict, and observability SHALL not expose raw tensor values, prompts, weights, cache contents, handles, or memory pointers by default.

#### Scenario: Tensor conversion planned

Given layout conversion is inserted

When observability records it

Then Runtime emits redacted conversion metadata.

---

### Requirement: Qwen Baseline Uses Tensor Contract

Qwen baseline graphs SHALL use Tensor Descriptors and Tensor Resources through
the Tensor Resource and Layout Contract.

#### Scenario: Qwen graph edge

Given Qwen prefill graph contains hidden state edge

When graph is validated

Then the edge has explicit shape, dtype, layout, and semantic role metadata.

---

### Requirement: Qwen Baseline Tensor Layout Is Explicit

Qwen baseline SHALL target explicit layouts and SHALL not assume hidden tensor
layout.

#### Scenario: Unknown layout

Given model tensor layout is unknown

When Qwen loading validates tensors

Then Runtime rejects or requires explicit materialization metadata.

---

### Requirement: Inference API Does Not Expose Raw Tensor Storage

Runtime Inference API SHALL not expose raw tensor storage, pointers, native handles, or Provider-owned opaque internals.

#### Scenario: Diagnostics include tensor

Given diagnostic includes tensor metadata

When Runtime returns it

Then only stable Tensor Resource metadata is included.

---

### Requirement: Inference API May Report Tensor Usage Metadata

Runtime Inference API SHALL report tensor usage metadata such as memory estimate or residency summary when policy allows.

#### Scenario: Usage report

Given usage report includes memory estimate

When caller receives it

Then it does not include raw tensor values or memory addresses.
